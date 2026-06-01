// Copyright(C) Facebook, Inc. and its affiliates.
use crate::{primary::PrimaryMessage, Header, Height};
use bytes::Bytes;
use config::Committee;
use crypto::{Digest, Hash, PublicKey};
use log::{debug, error, warn};
use network::{CancelHandler, ReliableSender, SimpleSender};
use store::Store;
use tokio::sync::mpsc::Receiver;

/// A task dedicated to help other authorities by replying to their certificates requests.
pub struct Helper {
    /// The committee information.
    committee: Committee,
    /// The persistent storage.
    store: Store,
    /// Input channel to receive certificates requests.
    rx_primaries_certs: Receiver<(Vec<Digest>, PublicKey)>,

    /// Input channel to receive certificates requests.
    rx_primaries_headers: Receiver<(Vec<Digest>, PublicKey)>,
    /// Input channel to receive fast sync header requests.
    rx_primaries_fast_sync_headers: Receiver<(Vec<(Digest, Height)>, PublicKey)>,
    /// A network sender to reply to the sync requests.
    //network: SimpleSender,
    // Change to reliable sender
    network: ReliableSender,
    // Cancel handlers
    cancel_handlers: Vec<CancelHandler>,
}

impl Helper {
    pub fn spawn(
        committee: Committee,
        store: Store,
        rx_primaries_certs: Receiver<(Vec<Digest>, PublicKey)>,
        rx_primaries_headers: Receiver<(Vec<Digest>, PublicKey)>,
        rx_primaries_fast_sync_headers: Receiver<(Vec<(Digest, Height)>, PublicKey)>,
    ) {
        tokio::spawn(async move {
            Self {
                committee,
                store,
                rx_primaries_certs,
                rx_primaries_headers,
                rx_primaries_fast_sync_headers,
                //network: SimpleSender::new(),
                network: ReliableSender::new(),
                cancel_handlers: Vec::new(),
            }
            .run()
            .await;
        });
    }

    async fn run(&mut self) {
        loop {
            tokio::select! {
                Some((digests, origin)) = self.rx_primaries_certs.recv() => {
                    // TODO [issue #195]: Do some accounting to prevent bad nodes from monopolizing our resources.

                    // get the requestors address.
                    let address = match self.committee.primary(&origin) {
                        Ok(x) => x.primary_to_primary,
                        Err(e) => {
                            warn!("Unexpected certificate request: {}", e);
                            continue;
                        }
                    };

                    // Reply to the request (the best we can).
                    for digest in digests {
                        match self.store.read(digest.to_vec()).await {
                            Ok(Some(data)) => {
                                // TODO: Remove this deserialization-serialization in the critical path.
                                let certificate = bincode::deserialize(&data)
                                    .expect("Failed to deserialize our own certificate");
                                let bytes = bincode::serialize(&PrimaryMessage::Certificate(certificate))
                                    .expect("Failed to serialize our own certificate");
                                let handler = self.network.send(address, Bytes::from(bytes)).await;
                                self.cancel_handlers.push(handler);
                            }
                            Ok(None) => (),
                            Err(e) => error!("{}", e),
                        }
                    }
                },
                Some((digests, origin)) = self.rx_primaries_headers.recv() => {
                    // TODO [issue #195]: Do some accounting to prevent bad nodes from monopolizing our resources.

                    // get the requestors address.
                    let address = match self.committee.primary(&origin) {
                        Ok(x) => x.primary_to_primary,
                        Err(e) => {
                            warn!("Unexpected certificate request: {}", e);
                            continue;
                        }
                    };

                    // Reply to the request (the best we can).
                    for digest in digests {
                        match self.store.read(digest.to_vec()).await {
                                Ok(Some(data)) => {
                                    //TODO: Remove this deserialization-serialization in the critical path.
                                    let header = bincode::deserialize(&data)
                                        .expect("Failed to deserialize our own certificate");
                                    let bytes = bincode::serialize(&PrimaryMessage::Header(header, true))  //sync = true
                                        .expect("Failed to serialize our own certificate");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                }
                                Ok(None) => (),
                                Err(e) => error!("{}", e),
                        }
                    }

                },
                Some((missing, origin)) = self.rx_primaries_fast_sync_headers.recv() => {
                    // TODO [issue #195]: Do some accounting to prevent bad nodes from monopolizing our resources.

                    // get the requestors address.
                    let address = match self.committee.primary(&origin) {
                        Ok(x) => x.primary_to_primary,
                        Err(e) => {
                            warn!("Unexpected certificate request: {}", e);
                            continue;
                        }
                    };

                    for (digest, lower_bound) in missing {
                        // Reply to the request (the best we can).
                        match self.store.read(digest.to_vec()).await {
                            Ok(Some(data)) => {
                                //TODO: Remove this deserialization-serialization in the critical path.
                                debug!("fast sync request handled success");
                                let header: Header = bincode::deserialize(&data)
                                    .expect("Failed to deserialize our own certificate");

                                let mut height = header.height() - 1;
                                let mut parent_digest = header.parent_cert.header_digest.clone();

                                let bytes = bincode::serialize(&PrimaryMessage::Header(header, true))  //sync = true
                                    .expect("Failed to serialize our own certificate");
                                let handler = self.network.send(address, Bytes::from(bytes)).await;
                                self.cancel_handlers.push(handler);

                                // Since we have the header in the store, we must have all of its ancestors
                                // Send sync replies for all ancestors until we reach the lower bound
                                while height > lower_bound {
                                    debug!("height is {}, lower bound is {}", height, lower_bound);
                                    let serialized_data = self.store.read(parent_digest.to_vec()).await.expect("should have ancestors").unwrap();
                                    let current_header: Header = bincode::deserialize(&serialized_data).expect("Failed to deserialize our own header");
                                    parent_digest = current_header.parent_cert.header_digest.clone();
                                    let bytes = bincode::serialize(&PrimaryMessage::Header(current_header, true))  //sync = true
                                        .expect("Failed to serialize our own header");
                                    let handler = self.network.send(address, Bytes::from(bytes)).await;
                                    self.cancel_handlers.push(handler);
                                    height -= 1;
                                }
                            }
                            Ok(None) => (),
                            Err(e) => error!("{}", e),
                        }
                    }
                },
            };
        }
    }
}
