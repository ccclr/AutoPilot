#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Copyright(C) Facebook, Inc. and its affiliates.
use crate::error::{DagError, DagResult};
use crate::messages::{proposal_digest, ConsensusMessage, Header, Proposal};
use crate::primary::{Height, PrimaryMessage, PrimaryWorkerMessage};
use bytes::Bytes;
use config::{Committee, WorkerId};
use crypto::{Digest, Hash, PublicKey};
use futures::future::try_join_all;
use futures::stream::futures_unordered::FuturesUnordered;
use futures::stream::StreamExt as _;
use log::{debug, error};
use network::{CancelHandler, ReliableSender};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use store::Store;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::{sleep, Duration, Instant};

/// The resolution of the timer that checks whether we received replies to our sync requests, and triggers
/// new sync requests if we didn't.
const TIMER_RESOLUTION: u64 = 1_000;

/// The commands that can be sent to the `Waiter`.
#[derive(Debug)]
pub enum WaiterMessage {
    SyncBatches(HashMap<Digest, WorkerId>, Header, bool),
    SyncProposals(Vec<(PublicKey, Proposal, Height)>, ConsensusMessage, Header),
    // SyncProposalsC(Vec<Proposal>, ConsensusMessage), //Consensus is independent of header.
    // SyncProposalsCAsync(Vec<Proposal>), //Consensus is independent of header.
    SyncParent(Digest, Header, u64),
    SyncHeader(Digest),
}

/// Waits for missing parent certificates and batches' digests.
pub struct HeaderWaiter {
    /// The name of this authority.
    name: PublicKey,
    /// The committee information.
    committee: Committee,
    /// The persistent storage.
    store: Store,
    /// The current consensus round (used for cleanup).
    consensus_round: Arc<AtomicU64>,
    /// The depth of the garbage collector.
    gc_depth: Height,
    /// The delay to wait before re-trying sync requests.
    sync_retry_delay: u64,
    /// Determine with how many nodes to sync when re-trying to send sync-request.
    sync_retry_nodes: usize,

    /// Receives sync commands from the `Synchronizer`.
    rx_synchronizer: Receiver<WaiterMessage>,
    /// Loops back to the core headers for which we got all parents and batches.
    tx_core: Sender<Header>,
    /// Loops back commit messages to the committer for reprocessing
    tx_consensus_loopback: Sender<(ConsensusMessage, Header)>,

    /// Network driver allowing to send messages.
    //network: SimpleSender,
    network: ReliableSender,

    /// Keeps the digests of the all certificates for which we sent a sync request,
    /// along with a timestamp (`u128`) indicating when we sent the request.
    parent_requests: HashMap<Digest, (Height, u128)>,
    //same, but for special parents
    header_requests: HashMap<Digest, (Height, u128)>,
    // whether fast sync is enabled
    use_fast_sync: bool,
    // whether optimistic tips is enabled
    use_optimistic_tips: bool,
    /// Keeps the digests of the all tx batches for which we sent a sync request,
    /// similarly to `header_requests`.
    batch_requests: HashMap<Digest, Height>,
    /// List of digests (either certificates, headers or tx batch) that are waiting
    /// to be processed. Their processing will resume when we get all their dependencies.
    pending: HashMap<Digest, (Height, Sender<()>)>,
    // Cancel handlers
    cancel_handlers: Vec<CancelHandler>,
}

impl HeaderWaiter {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        name: PublicKey,
        committee: Committee,
        store: Store,
        consensus_round: Arc<AtomicU64>,
        gc_depth: Height,
        sync_retry_delay: u64,
        sync_retry_nodes: usize,
        rx_synchronizer: Receiver<WaiterMessage>,
        tx_core: Sender<Header>,
        tx_consensus_loopback: Sender<(ConsensusMessage, Header)>,
        use_fast_sync: bool,
        use_optimistic_tips: bool,
    ) {
        tokio::spawn(async move {
            Self {
                name,
                committee: committee.clone(),
                store,
                consensus_round,
                gc_depth,
                sync_retry_delay,
                sync_retry_nodes,
                rx_synchronizer,
                tx_core,
                tx_consensus_loopback,
                //network: SimpleSender::new(),
                network: ReliableSender::new(),
                parent_requests: HashMap::new(),
                header_requests: HashMap::new(),
                batch_requests: HashMap::new(),
                cancel_handlers: Vec::new(),
                use_fast_sync,
                use_optimistic_tips,
                pending: HashMap::new(),
            }
            .run()
            .await;
        });
    }

    /// Helper function. It waits for particular data to become available in the storage
    /// and then delivers the specified header.
    async fn waiter(
        mut missing: Vec<(Vec<u8>, Store)>,
        deliver: Header,
        mut handler: Receiver<()>,
    ) -> DagResult<Option<Header>> {
        let waiting: Vec<_> = missing
            .iter_mut()
            .map(|(x, y)| y.notify_read(x.to_vec()))
            .collect();
        tokio::select! {
            result = try_join_all(waiting) => {
                result.map(|_| Some(deliver)).map_err(DagError::from)
            }
            _ = handler.recv() => Ok(None),
        }
    }

    async fn proposal_waiter(
        mut missing: Vec<(Vec<u8>, Store)>,
        deliver: (ConsensusMessage, Header),
        mut handler: Receiver<()>,
    ) -> DagResult<Option<(ConsensusMessage, Header)>> {
        let waiting: Vec<_> = missing
            .iter_mut()
            .map(|(x, y)| y.notify_read(x.to_vec()))
            .collect();
        tokio::select! {
            result = try_join_all(waiting) => {
                result.map(|_| Some(deliver)).map_err(DagError::from)
            }
            _ = handler.recv() => Ok(None),
        }
    }

    async fn optimistic_tip_waiter(
        mut missing: Vec<(Vec<u8>, Store)>,
        deliver: (ConsensusMessage, Header),
        mut handler: Receiver<()>,
    ) -> DagResult<Option<(ConsensusMessage, Header)>> {
        let waiting: Vec<_> = missing
            .iter_mut()
            .map(|(x, y)| y.notify_read(x.to_vec()))
            .collect();
        tokio::select! {
            result = try_join_all(waiting) => {
                result.map(|_| Some(deliver)).map_err(DagError::from)
            }
            _ = handler.recv() => Ok(None),
        }
    }

    /// Main loop listening to the `Synchronizer` messages.
    async fn run(&mut self) {
        let mut waiting = FuturesUnordered::new();
        let mut proposal_waiting = FuturesUnordered::new();
        let mut prepare_proposal_waiting = FuturesUnordered::new();

        let timer = sleep(Duration::from_millis(TIMER_RESOLUTION));
        tokio::pin!(timer);

        loop {
            tokio::select! {
                Some(message) = self.rx_synchronizer.recv() => {
                    match message {
                        WaiterMessage::SyncBatches(missing, header, force_sync) => {
                            debug!("Synching the payload of {}", header);
                            debug!("Missing payloads are {:?}", missing);
                            let header_id = header.id.clone();
                            let header_id1 = header.id.clone();
                            let round = header.height;
                            let author = header.author;

                            // Ensure we sync only once per header.
                            if self.pending.contains_key(&header_id) {
                                debug!("already pending for header {}", header_id);
                                continue;
                            }

                            // Add the header to the waiter pool. The waiter will return it to when all
                            // its parents are in the store.
                            let wait_for = missing
                                .iter()
                                .map(|(digest, worker_id)| {
                                    let key = [digest.as_ref(), &worker_id.to_le_bytes()].concat();
                                    (key.to_vec(), self.store.clone())
                                })
                                .collect();
                            let (tx_cancel, rx_cancel) = channel(1);
                            self.pending.insert(header_id.clone(), (round, tx_cancel));
                            debug!("SyncBatches pending insert for header {}", header_id);
                            let fut = Self::waiter(wait_for, header, rx_cancel);
                            waiting.push(fut);

                            if true {
                            //if force_sync {
                                // Ensure we didn't already send a sync request for these parents.
                                let mut requires_sync = HashMap::new();
                                for (digest, worker_id) in missing.into_iter() {
                                    self.batch_requests.entry(digest.clone()).or_insert_with(|| {
                                        requires_sync.entry(worker_id).or_insert_with(Vec::new).push(digest);
                                        round
                                    });
                                }
                                debug!("requires_sync is {:?} for header {:?}", requires_sync, header_id1);
                                for (worker_id, digests) in requires_sync {
                                    let address = self.committee
                                        .worker(&self.name, &worker_id)
                                        .expect("Author of valid header is not in the committee")
                                        .primary_to_worker;
                                    debug!("Sent syncbatches message for height {}, digests {:?}", round, digests);

                                    let message = PrimaryWorkerMessage::Synchronize(digests, author);
                                    let bytes = bincode::serialize(&message)
                                        .expect("Failed to serialize batch sync request");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                    //self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;
                                }
                            }
                        }

                        WaiterMessage::SyncHeader(missing) => {
                            debug!("Syncing on header with digest {}", missing);

                            let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Failed to measure time")
                            .as_millis();

                            let mut requires_sync = Vec::new();
                            self.header_requests.entry(missing.clone()).or_insert_with(|| {
                                requires_sync.push(missing);
                                (0, now)
                            });

                            if !requires_sync.is_empty() {
                                let addresses = self.committee
                                .others_primaries(&self.name)
                                .iter()
                                .map(|(_, x)| x.primary_to_primary)
                                .collect();

                                let message = PrimaryMessage::HeadersRequest(requires_sync, self.name);
                                let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                                self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await; //after timeout, re-broadcast again (technically not necessary)
                            }
                        }

                        WaiterMessage::SyncParent(missing, header, lower_bound) => {
                            debug!("Synching the parents of {}", header);
                            let header_id = header.id.clone();
                            let height = header.height();
                            let author = header.author;

                            // Ensure we sync only once per header.
                            if self.pending.contains_key(&header_id) {
                                continue;
                            }

                            debug!("use fast sync is {}", self.use_fast_sync);
                            debug!("parent missing digest is {:?}", missing);

                            // Add the header to the waiter pool. The waiter will return it to us
                            // when all its parents are in the store.
                            let mut wait_for = Vec::new();
                            wait_for.push((missing.to_vec(), self.store.clone()));
                            let (tx_cancel, rx_cancel) = channel(1);
                            self.pending.insert(header_id.clone(), (height, tx_cancel));
                            debug!("SyncParent pending insert for header {}", header_id);
                            let fut = Self::waiter(wait_for, header, rx_cancel);
                            waiting.push(fut);

                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Failed to measure time")
                                .as_millis();



                            // Check whether we should send a fast sync request to the network to avoid duplicate sync requests
                            if self.use_fast_sync {
                                let should_sync = height - 1 > lower_bound;
                                if should_sync {
                                    debug!("send a fast sync parent request with height {}, lower bound {}", height, lower_bound);
                                    let mut requires_sync = Vec::new();

                                    self.parent_requests.entry(missing.clone()).or_insert_with(|| {
                                        requires_sync.push((missing.clone(), lower_bound));
                                        (lower_bound, now)
                                    });
                                    let address = self.committee
                                        .primary(&author)
                                        .expect("Author of valid header not in the committee")
                                        .primary_to_primary;
                                    // Lower bound to stop syncing is last height
                                    let message = PrimaryMessage::FastSyncHeadersRequest(requires_sync, self.name);
                                    let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                } else {
                                    debug!("already sent fast sync request do not send duplicate");
                                }
                            } else {
                                // Ensure we didn't already sent a sync request for these parents.
                                // Optimistically send the sync request to the node that created the certificate.
                                // If this fails (after a timeout), we broadcast the sync request.
                                let mut requires_sync = Vec::new();
                                self.parent_requests.entry(missing.clone()).or_insert_with(|| {
                                    requires_sync.push(missing);
                                    (height, now)
                                });
                                if !requires_sync.is_empty() {
                                    let address = self.committee
                                        .primary(&author)
                                        .expect("Author of valid header not in the committee")
                                        .primary_to_primary;
                                    let message = PrimaryMessage::HeadersRequest(requires_sync, self.name);
                                    let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);

                                }
                            }
                        }


                        WaiterMessage::SyncProposals(missing, consensus_message, header) => {
                            //let header_id = header.id.clone();
                            let height = header.height();
                            let author = header.author;
                            let id = proposal_digest(&consensus_message);
                            //println!("syncing proposals in header waiter");
                            let mut pending_already_set = false;

                            // Ensure we sync only once per proposal
                            if self.pending.contains_key(&id) {
                                pending_already_set = true;
                                //continue;
                            }

                            debug!("use fast sync is {}", self.use_fast_sync);

                            // If optimistic tips enabled and it's a prepare message, use the optimistic tip waiter
                            match consensus_message {
                                ConsensusMessage::Prepare { slot, view: _, tc: _, qc_ticket: _, proposals: _, aggregate_report: _} => {
                                    if self.use_optimistic_tips {
                                        // Add the header to the waiter pool. The waiter will return it to us
                                        // when all its parents are in the store.
                                        let wait_for_opt = missing
                                            .iter()
                                            .cloned()
                                            .map(|(_, x, _)| ({let mut opt_key = x.header_digest.to_vec(); opt_key.push(1); debug!("opt key is {:?}", opt_key); opt_key}, self.store.clone()))
                                            .collect();
                                        let (tx_cancel, rx_cancel) = channel(1);
                                        self.pending.insert(id.clone(), (height, tx_cancel));
                                        debug!("SyncProposals pending insert for header {}", id);
                                        let fut = Self::optimistic_tip_waiter(wait_for_opt, (consensus_message, header), rx_cancel);
                                        debug!("adding waiter for optimistic tip");
                                        prepare_proposal_waiting.push(fut);
                                    } else {
                                        // Add the header to the waiter pool. The waiter will return it to us
                                        // when all its parents are in the store.
                                        let wait_for = missing
                                            .iter()
                                            .cloned()
                                            .map(|(_, x, _)| (x.header_digest.to_vec(), self.store.clone()))
                                            .collect();
                                        let (tx_cancel, rx_cancel) = channel(1);
                                        self.pending.insert(id.clone(), (height, tx_cancel));
                                        debug!("SyncProposals pending insert for header {}", id);
                                        let fut = Self::proposal_waiter(wait_for, (consensus_message, header), rx_cancel);
                                        //println!("created proposal waiter");
                                        debug!("normal proposal waiter");
                                        proposal_waiting.push(fut);
                                    }
                                },
                                _ => {
                                    // Add the header to the waiter pool. The waiter will return it to us
                                    // when all its parents are in the store.
                                    let wait_for = missing
                                        .iter()
                                        .cloned()
                                        .map(|(_, x, _)| (x.header_digest.to_vec(), self.store.clone()))
                                        .collect();
                                    let (tx_cancel, rx_cancel) = channel(1);
                                    self.pending.insert(id.clone(), (height, tx_cancel));
                                    debug!("SyncProposals pending insert for header {}", id);
                                    let fut = Self::proposal_waiter(wait_for, (consensus_message, header), rx_cancel);
                                    debug!("normal proposal waiter");
                                    //println!("created proposal waiter");
                                    proposal_waiting.push(fut);
                                }
                            }

                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Failed to measure time")
                                .as_millis();

                            // Check whether we should send a fast sync request to the network to avoid duplicate sync requests
                            if self.use_fast_sync {
                                let mut requires_sync = Vec::new();

                                for (pk, proposal, lower_bound) in missing {
                                    debug!("send a fast sync proposal request with height {}, lower bound {}", proposal.height, lower_bound);
                                    debug!("opt digest sync is {:?}", proposal.header_digest);

                                    let should_sync = proposal.height > lower_bound;

                                    if should_sync && !pending_already_set {
                                        self.parent_requests.entry(proposal.header_digest.clone()).or_insert_with(|| {
                                            requires_sync.push((proposal.header_digest, lower_bound));
                                            (lower_bound, now)
                                        });
                                    } else {
                                        debug!("already sent fast sync request do not send duplicate");
                                    }
                                }

                                if !requires_sync.is_empty() {
                                    let address = self.committee
                                        .primary(&author)
                                        .expect("Author of valid header not in the committee")
                                        .primary_to_primary;
                                    let message = PrimaryMessage::FastSyncHeadersRequest(requires_sync, self.name);
                                    let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                    /*let addresses = self.committee.others_primaries(&self.name).iter().map(|(_, x)| x.primary_to_primary).collect();
                                    self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;*/

                                }
                            } else {
                                // Ensure we didn't already sent a sync request for these parents.
                                // Optimistically send the sync request to the node that created the certificate.
                                // If this fails (after a timeout), we broadcast the sync request.

                                let mut requires_sync = Vec::new();
                                for (_, missing, _) in missing {
                                    self.parent_requests.entry(missing.header_digest.clone()).or_insert_with(|| {
                                        requires_sync.push(missing.header_digest);
                                        (missing.height, now)
                                    });
                                }

                                if !requires_sync.is_empty() {
                                    let address = self.committee
                                        .primary(&author)
                                        .expect("Author of valid header not in the committee")
                                        .primary_to_primary;
                                    let message = PrimaryMessage::HeadersRequest(requires_sync, self.name);
                                    let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                    /*let addresses = self.committee.others_primaries(&self.name).iter().map(|(_, x)| x.primary_to_primary).collect();
                                    self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;*/
                                }
                            }
                        }
                    }
                },

                Some(result) = waiting.next() => match result {
                    Ok(Some(header)) => {
                        debug!("Finished synching digest {:?} for header {:?}", header.parent_cert.header_digest, header);
                        let _ = self.pending.remove(&header.id);
                        for x in header.payload.keys() {
                            let _ = self.batch_requests.remove(x);
                        }
                        let _ = self.parent_requests.remove(&header.parent_cert.header_digest);

                        self.tx_core.send(header).await.expect("Failed to send header");
                    },
                    Ok(None) => {
                        // This request has been canceled.
                    },
                    Err(e) => {
                        error!("{}", e);
                        panic!("Storage failure: killing node.");
                    }
                },

                Some(result) = proposal_waiting.next() => match result {
                    Ok(Some(deliver)) => {
                        //println!("finished syncing");
                        let id = proposal_digest(&deliver.0);
                        let _ = self.pending.remove(&id);
                        //let _ = self.cancel_handlers.remove(&id);
                        for x in deliver.1.payload.keys() {
                            let _ = self.batch_requests.remove(x);
                        }

                        let possibly_missing;
                        match &deliver.0 {
                            ConsensusMessage::Prepare {view: _, slot: _, tc: _, qc_ticket: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                            ConsensusMessage::Confirm {view: _, slot: _, qc: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                            ConsensusMessage::Commit {view: _, slot: _, qc: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                        }
                        for (_, prop) in possibly_missing.iter() {
                            debug!("removing prop digest {:?}", prop.header_digest);
                            let _ = self.parent_requests.remove(&prop.header_digest);
                        }

                        debug!("wake up normal proposals");
                        self.tx_consensus_loopback.send(deliver).await.expect("Failed to send header");
                    },
                    Ok(None) => {
                        // This request has been canceled.
                    },
                    Err(e) => {
                        error!("{}", e);
                        panic!("Storage failure: killing node.");
                    }
                },

                Some(result) = prepare_proposal_waiting.next() => match result {
                    Ok(Some(deliver)) => {
                        //println!("finished syncing");
                        let id = proposal_digest(&deliver.0);
                        let _ = self.pending.remove(&id);
                        //let _ = self.cancel_handlers.remove(&id);
                        for x in deliver.1.payload.keys() {
                            let _ = self.batch_requests.remove(x);
                        }

                        let possibly_missing;
                        match &deliver.0 {
                            ConsensusMessage::Prepare {view: _, slot: _, tc: _, qc_ticket: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                            ConsensusMessage::Confirm {view: _, slot: _, qc: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                            ConsensusMessage::Commit {view: _, slot: _, qc: _, proposals, aggregate_report: _} => {possibly_missing = proposals},
                        }
                        for (_, prop) in possibly_missing.iter() {
                            debug!("removing prop digest {:?}", prop.header_digest);
                            let _ = self.parent_requests.remove(&prop.header_digest);
                        }
                        debug!("wake up prepare optimistic tips");
                        self.tx_consensus_loopback.send(deliver).await.expect("Failed to send header");
                    },
                    Ok(None) => {
                        // This request has been canceled.
                    },
                    Err(e) => {
                        error!("{}", e);
                        panic!("Storage failure: killing node.");
                    }
                },

                () = &mut timer => {
                    // We optimistically sent sync requests to a single node. If this timer triggers,
                    // it means we were wrong to trust it. We are done waiting for a reply and we now
                    // broadcast the request to all nodes.
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Failed to measure time")
                        .as_millis();

                    //Retry HeaderRequests  -- We don't use this
                    // let mut retry = Vec::new();
                    // for (digest, (_, timestamp)) in &self.header_requests {
                    //     if timestamp + (self.sync_retry_delay as u128) < now {
                    //         debug!("Requesting sync for header {} (retry)", digest);
                    //         retry.push(digest.clone());
                    //     }
                    // }
                    // let addresses = self.committee.others_primaries(&self.name).iter().map(|(_, x)| x.primary_to_primary).collect();
                    // let message = PrimaryMessage::HeadersRequest(retry, self.name);
                    // let bytes = bincode::serialize(&message).expect("Failed to serialize header request");
                    // self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;

                    //Retry CertificateRequests
                    let mut retry = Vec::new();
                    let mut retry_fast_sync = Vec::new();

                    for (digest, (lower_bound, timestamp)) in &self.parent_requests {
                        if timestamp + (self.sync_retry_delay as u128) < now {
                            debug!("Requesting retry sync for parent header {} (retry)", digest);
                            if self.use_fast_sync {
                                retry_fast_sync.push((digest.clone(), *lower_bound));
                            } else {
                                retry.push(digest.clone());
                            }
                        }
                    }
                    let addresses = self.committee.others_primaries(&self.name).iter().map(|(_, x)| x.primary_to_primary).collect();
                    if self.use_fast_sync {
                        let message = PrimaryMessage::FastSyncHeadersRequest(retry_fast_sync, self.name);
                        let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                        let handlers = self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;
                        self.cancel_handlers.extend(handlers);

                    } else {
                        let message = PrimaryMessage::HeadersRequest(retry, self.name);
                        let bytes = bincode::serialize(&message).expect("Failed to serialize cert request");
                        let handlers = self.network.lucky_broadcast(addresses, Bytes::from(bytes), self.sync_retry_nodes).await;

                        self.cancel_handlers.extend(handlers);
                    }

                    // Reschedule the timer.
                    timer.as_mut().reset(Instant::now() + Duration::from_millis(TIMER_RESOLUTION));
                }
            }

            // Cleanup internal state.
            let round = self.consensus_round.load(Ordering::Relaxed);
            if round > self.gc_depth {
                let mut gc_round = round - self.gc_depth;

                for (r, handler) in self.pending.values() {
                    if r <= &gc_round {
                        let _ = handler.send(()).await;
                    }
                }
                /*self.pending.retain(|_, (r, _)| r > &mut gc_round);
                self.batch_requests.retain(|_, r| r > &mut gc_round);
                self.parent_requests.retain(|_, (r, _)| r > &mut gc_round);
                self.header_requests.retain(|_, (r, _)| r > &mut gc_round);*/
            }
        }
    }
}
