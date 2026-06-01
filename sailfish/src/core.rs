use std::convert::TryInto;
use std::net::SocketAddr;
use crate::aggregator::Aggregator;
use crate::consensus::{ConsensusMessage, View, Round, Slot};
//use crate::error::{ConsensusError, ConsensusResult};
use crate::leader::LeaderElector;
use crate::mempool::MempoolDriver;
//use crate::messages::{Header, Certificate, Block, Timeout, Vote, AcceptVote, QC, TC};
//use crate::proposer::ProposerMessage;
use crate::synchronizer::Synchronizer;
use crate::timer::Timer;
use async_recursion::async_recursion;
use bytes::Bytes;
use config::{Committee, Stake};
use crypto::{Hash as _, CryptoError, Signature};
use crypto::{PublicKey, SignatureService, Digest};
use futures::FutureExt;
use log::{debug, error, warn};
use network::SimpleSender;
use primary::ensure;
use primary::messages::{Header, Certificate, Timeout, TC, Ticket, Vote, Info, CertType, PrepareInfo, ConfirmInfo, InstanceInfo};
use primary::error::{ConsensusError, ConsensusResult};
use std::cmp::max;
use std::collections::{VecDeque, HashSet, BTreeMap};
use std::collections::HashMap;
use store::Store;
use tokio::sync::mpsc::{Receiver, Sender};
use ed25519_dalek::Sha512;
use ed25519_dalek::Digest as _;


#[cfg(test)]
#[path = "tests/core_tests.rs"]

pub mod core_tests;

pub struct Core {
    name: PublicKey,
    committee: Committee,
    store: Store,
    signature_service: SignatureService,
    leader_elector: LeaderElector,
    mempool_driver: MempoolDriver,
    synchronizer: Synchronizer,
    rx_message: Receiver<ConsensusMessage>,
    rx_consensus: Receiver<Certificate>,
    rx_loopback_header: Receiver<Header>,
    rx_loopback_cert: Receiver<Certificate>,
    tx_pushdown_cert: Sender<Certificate>,
    //tx_proposer: Sender<ProposerMessage>,
    tx_commit: Sender<Certificate>,
    tx_validation: Sender<(Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>)>,
    tx_info: Sender<PrepareInfo>, //Sender<(View, Round, Ticket)>,
    rx_special: Receiver<Header>,
    rx_loopback_process_commit: Receiver<(Digest, Ticket)>,
    rx_loopback_commit: Receiver<(Header, Ticket)>,

    view: View,
    last_prepared_view: View,
    last_voted_view: View,
    last_committed_view: View,
    stored_headers: HashMap<Digest, Header>,
    high_prepare: Header,
    high_accept: Certificate,  //this is a DAG QC
    high_tc: TC,             //last TC --> just in case we need to reply with a proof. ::Note: This is not necessary if we are not proving invalidations. (For Timeout messages high_prepare suffices)
    timer: Timer,
    aggregator: Aggregator,
    network: SimpleSender,
    round: Round,
   
    tips: HashMap<PublicKey, Header>,
    current_certs: HashMap<PublicKey, Certificate>,
    views: HashMap<Slot, View>,
    timers: HashMap<Slot, Timer>,
    qcs: HashMap<Slot, Info>,
    tcs: HashMap<Slot, TC>,
    ticket: Ticket,
    already_proposed_slots: Vec<Slot>,
    committed: HashMap<Slot, ConfirmInfo>,


    processing_headers: HashMap<View, HashSet<Digest>>,
    processing_certs: HashMap<View, HashSet<Digest>>,
    // process_commit: HashSet<Digest>, //Digest of the Header ready to commit -- all consensus parents are available
    // waiting_to_commit:  HashSet<Digest>, // Digest of the Header thats eligible to commit (has QC), but not ready because cert has not been processed
}

impl Core {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        name: PublicKey,
        committee: Committee,
        signature_service: SignatureService,
        store: Store,
        leader_elector: LeaderElector,
        mempool_driver: MempoolDriver,
        synchronizer: Synchronizer,
        timeout_delay: u64,
        rx_message: Receiver<ConsensusMessage>,
        rx_consensus: Receiver<Certificate>,
        rx_loopback_header: Receiver<Header>,
        rx_loopback_cert: Receiver<Certificate>,
        tx_pushdown_cert: Sender<Certificate>,
        tx_commit: Sender<Certificate>,
        tx_validation: Sender<(Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>)>, // Loopback Special Headers to DAG
        tx_info: Sender<PrepareInfo>, 
        rx_special: Receiver<Header>,
        rx_loopback_process_commit: Receiver<(Digest, Ticket)>,
        rx_loopback_commit: Receiver<(Header, Ticket)>,
    ) {
        tokio::spawn(async move {
            Self {
                name,
                committee: committee.clone(),
                signature_service,
                store,
                leader_elector,
                mempool_driver,
                synchronizer,
                rx_message,
                rx_consensus,
                rx_loopback_header,
                rx_loopback_cert,
                tx_pushdown_cert,
                //tx_proposer,
                tx_commit,
                tx_validation,
                tx_info,
                rx_special,
                rx_loopback_process_commit,
                rx_loopback_commit,

                view: 1,
                last_prepared_view: 0, 
                last_voted_view: 0,
                last_committed_view: 0,
                stored_headers: HashMap::default(),         //TODO: Can remove?
                high_prepare: Header::genesis(&committee),
                high_accept: Certificate::genesis_cert(&committee), //== first cert of Cert::genesis
                high_tc: TC::genesis(&committee),
                timer: Timer::new(timeout_delay),
                aggregator: Aggregator::new(committee),
                network: SimpleSender::new(),
                round: 0,
                processing_headers: HashMap::new(),
                processing_certs: HashMap::new(),
                tips: HashMap::new(),
                current_certs: HashMap::new(),
                views: HashMap::new(),
                timers: HashMap::new(),
                qcs: HashMap::new(),
                tcs: HashMap::new(),
                ticket: Ticket { header: None, tc: None, slot: 1, proposals: BTreeMap::new() },
                already_proposed_slots: Vec::new(),
                committed: HashMap::new(),
                // process_commit: HashSet::new(),
                // waiting_to_commit: HashSet::new(),
            }
            .run()
            .await
        });
    }

    async fn store_header(&mut self, header: &Header) {
        let key = header.id.to_vec();
        let value = bincode::serialize(header).expect("Failed to serialize header");
        self.store.write(key, value).await;
    }

    async fn store_cert(&mut self, cert: &Certificate) {
        let key = cert.digest().to_vec();
        let value = bincode::serialize(cert).expect("Failed to serialize cert");
        self.store.write(key, value).await;
    }


    fn increase_last_prepared_view(&mut self, target: View) {
        self.last_prepared_view = max(self.last_prepared_view, target);
    }

    fn increase_last_voted_view(&mut self, target: View) {
        self.last_voted_view = max(self.last_voted_view, target);
    }

    /*async fn generate_ticket_and_commit(&mut self, header_digest: Digest, header: Option<Header>, qc: Option<QC>, tc: Option<TC>) -> ConsensusResult<Ticket> {
      
        //generate header
        //let new_ticket = Ticket::new(qc.unwrap(), tc, header_digest.clone(), self.view).await;
        
        //Create new ticket, using the associated Qc/Tc and it's view  (Important: Don't use latest view)
        let new_ticket = match tc {
            Some(timeout_cert) => {
                Ticket::new(header_digest.clone(), timeout_cert.view.clone(), timeout_cert.view_round.clone(), self.high_qc.clone(), Some(timeout_cert)).await
            },
            None => {
                match qc {
                    Some(quorum_cert) => Ticket::new(header_digest.clone(), quorum_cert.view.clone(), quorum_cert.view_round.clone(), quorum_cert.clone(), None).await,
                    None => return Err(ConsensusError::InvalidTicket),
                }
            }
           
        };

        if let Some(header) = header {
            self.commit(header, new_ticket.clone()).await?;
        }
        else{
            self.process_commit(header_digest, new_ticket.clone()).await?; //TODO: Can execute this asynchronously? 
        }
        
        Ok(new_ticket)
    }*/

    /*async fn process_commit(&mut self, header_digest: Digest, new_ticket: Ticket) -> ConsensusResult<()> {
        //Fetch header. If not available, sync first.
        match self.synchronizer.get_commit_header(header_digest, &new_ticket).await? { 
            // If have header, call commit
           Some(header) => self.commit(header, new_ticket).await?,
            // If don't have header, start sync. On reply call commit.  (Same reply as the ticket sync reply!!) ==> both use same sync function.
           None => {debug!("don't have header");}
       }
       Ok(())
    }*/

    /*async fn commit(&mut self, header: Header, new_ticket: Ticket) -> ConsensusResult<()> {
        //TODO: Need to handle duplicate calls?
        if self.last_committed_view >= header.view {
            return Ok(());
        }

        // deliver_parent_ticket.  confirm that ticket is in store. Else create waiter to re-trigger commit. //sync on ticket header (and directly call commit on it once received)
        if !self.synchronizer.deliver_parent_ticket(&header, &new_ticket).await? {
            return Ok(());
        }

        //advance last round to ticket round.
        self.round = max(self.round, new_ticket.round);

        if new_ticket.tc.is_some() { //don't commit it, just store it.
            // store the ticket.
            let bytes = bincode::serialize(&new_ticket).expect("Failed to serialize ticket");
            self.store.write(new_ticket.digest().to_vec(), bytes).await;
            return Ok(());
        }
    
        //adopt header view
        self.last_committed_view = header.view.clone();
        
        self.processing_headers.retain(|k, _| k >= &header.view); //garbage collect
        self.processing_certs.retain(|k, _| k >= &header.view); //garbage collect

        let mut to_commit = VecDeque::new();
        let mut parent = header;
        to_commit.push_back(parent.clone());

       //Commits parents too. This will only happen in case of View Changes, i.e. if the parent ticket is a TC. 
        while self.last_committed_view + 1 < parent.view {
            let ancestor:Header = self
                .synchronizer
                .get_parent_header(&parent) //TODO: Create. Lookup consensus parent ticket (should be there), and then the header from that.
                .await?
                .expect("Should have ancestor by now");
            to_commit.push_back(ancestor.clone());
            parent= ancestor;
        }

        

        let mut certs = Vec::new();
         // Send all the newly committed blocks to the node's application layer.
         while let Some(header) = to_commit.pop_back() {
            debug!("Committed {:?}", header);
            
            let cert = Certificate { header, special_valids: Vec::new(), votes: Vec::new() };
            certs.push(cert.clone());
            self.tx_commit   //Note: This is sent to the commiter->CertificateWaiter. It waits for all Dag parent certs to arrive. However, this is already guaranteed by our implicit requirement that the committing cert is processed.
                .send(cert)
                .await
                .expect("Failed to send commit cert to committer");
            
            // Cleanup the mempool.
            
        }
        self.mempool_driver.cleanup(certs).await;

        //store it store the ticket.
        let bytes = bincode::serialize(&new_ticket).expect("Failed to serialize ticket");
        self.store.write(new_ticket.digest().to_vec(), bytes).await;
        Ok(())
    }*/

    // async fn commit(&mut self, block: Block, qc: QC) -> ConsensusResult<()> {
    //     if self.last_committed_view >= block.view {
    //         return Ok(());
    //     }

    //     let mut to_commit = VecDeque::new();
    //     to_commit.push_back(block.clone());

    //     // Ensure we commit the entire chain. This is needed after view-change.
    //     let mut parent = block.clone();
    //     while self.last_committed_view + 1 < parent.view {
    //         let ancestor = self
    //             .synchronizer
    //             .get_parent_block(&parent)
    //             .await?
    //             .expect("We should have all the ancestors by now");
    //         to_commit.push_back(ancestor.clone());
    //         parent = ancestor;
    //     }

    //     // Save the last committed block.
    //     self.last_committed_view = block.view;

    //     // Send all the newly committed blocks to the node's application layer.
    //     while let Some(block) = to_commit.pop_back() {
    //         debug!("Committed {:?}", block);

    //         // Output the block to the top-level application.
    //         if let Err(e) = self.tx_output.send(block.clone()).await {
    //             warn!("Failed to send block through the output channel: {}", e);
    //         }

    //         let mut certs = Vec::new();
    //         // Send the payload to the committer.
    //         for header in block.payload {
    //             let certificate = Certificate { header, votes: qc.clone().votes };
    //             certs.push(certificate.clone());

    //             self.tx_commit
    //                 .send(certificate)
    //                 .await
    //                 .expect("Failed to send payload");
    //         }

    //         // Cleanup the mempool.
    //         self.mempool_driver.cleanup(certs).await;
    //     }
    //     Ok(())
    // }


    /*async fn local_timeout_view(&mut self) -> ConsensusResult<()> {
       //TESTING 
        return Ok(());

        warn!("Timeout reached for view {}", self.view);
        //println!("timeout reached for view {}", self.view);

        // Increase the last prepared and voted view.
        self.increase_last_prepared_view(self.view);
        self.increase_last_voted_view(self.view);

        let mut send_high_prepare = false;
        let mut send_high_accept = false;
        let mut send_high_qc = false;
        //Send either high_qc or high_accept depending on which is newer (both suffice to commit)
        if self.high_qc.view >= self.high_accept.header.view {
            send_high_qc = true;
        }
        else {
            send_high_accept = true
        }
        //Send High Prepare only if it is newer
        if self.high_prepare.view > max(self.high_qc.view, self.high_accept.header.view) {
            send_high_prepare = true;
        }
      
        // Make a timeout message.
        let timeout = Timeout::new(
            if send_high_prepare {Some(self.high_prepare.clone())} else {None},
            if send_high_accept {Some(self.high_accept.clone())} else {None},
            if send_high_qc {Some(self.high_qc.clone())} else {None},
            self.view,
            self.name,
            self.signature_service.clone(),
        )
        .await;
        debug!("Created {:?}", timeout);

        // Reset the timer.
        self.timer.reset();

        // Broadcast the timeout message.
        debug!("Broadcasting {:?}", timeout);
        let addresses: Vec<SocketAddr> = self
            .committee
            .others_consensus(&self.name)
            .into_iter()
            .map(|(_, x)| x.consensus_to_consensus)
            .collect();
        let message = bincode::serialize(&ConsensusMessage::Timeout(timeout.clone()))
            .expect("Failed to serialize timeout message");
        //println!("timeout message bytes {:?}", message.clone());
        self.network
            .broadcast(addresses.clone(), Bytes::from(message.clone()))
            .await;
        //self.network.broadcast(addresses, Bytes::from(message)).await;

        // Process our message.
        self.handle_timeout(timeout).await
    }

    async fn handle_timeout(&mut self, timeout: Timeout) -> ConsensusResult<()> {
        debug!("Processing {:?}", timeout);
        if timeout.view < self.view {
            return Ok(());
        }

        // Ensure the timeout is well formed.
        timeout.verify(&self.committee)?;


        // Handle the QC embedded in the timeout if present.
        if let Some(qc) = &timeout.high_qc {
            let res = self.handle_qc(qc.clone()).await;
        }
       
        //TODO: NOTE: Don't view change just because we receive one timeout (could be from byz) ==> only accept view change once >= f+1 arrive. Waiting for TC to form (>= 2f+1) is also fine.

        // Add the new vote to our aggregator and see if we have a quorum.
        if let Some(tc) = self.aggregator.add_timeout(timeout)? {  //TODO: FIXME: Re-factor this to be an aggregator per view. Garbage collect old ones.
            debug!("Assembled {:?}", tc);

            //The TC will be forwarded as part of the next proposals ticket. If we are pessimistic/we want it faster, we can also broadcast
            let pessimistic_tc_exchange = true; //TODO: Turn this into a flag.
              // Broadcast the QC. //alternatively: pass it down as ticket.
            if pessimistic_tc_exchange {
                // Broadcast the TC.
                debug!("Broadcasting {:?}", tc);
                let addresses = self
                    .committee
                    .others_consensus(&self.name)
                    .into_iter()
                    .map(|(_, x)| x.consensus_to_consensus)
                    .collect();
                let message = bincode::serialize(&ConsensusMessage::TC(tc.clone()))
                    .expect("Failed to serialize timeout certificate");
                self.network
                    .broadcast(addresses, Bytes::from(message))
                    .await;
            }

            self.process_tc(tc).await?;

        }
        Ok(())
    }*/

    #[async_recursion]
    async fn advance_view(&mut self, view: View) {
        if view < self.view {
            return;
        }
        // Reset the timer and advance view.
        self.timer.reset();
        self.view = view + 1;
        debug!("Moved to view {}", self.view);

        // Cleanup the vote aggregator.
        self.aggregator.cleanup(&self.view);
    }

    // TODO: Add actual validity checks
    fn is_prepare_valid(&mut self, info: PrepareInfo) -> bool {
        true
    }

    // TODO: Add actual validity checks
    fn is_confirm_valid(&mut self, info: ConfirmInfo) -> bool {
        true
    }

    async fn process_prepare_info(&mut self, header: Header, info: PrepareInfo, instance_info: InstanceInfo) {
        let prepare_valid = self.is_prepare_valid(info.clone());
        if prepare_valid {
            //self.views[&info.consensus_info.slot] = info.consensus_info.view;
            //self.timers[info.consensus_info.slot] = Timer::new(5);

            let new_instance_info: InstanceInfo = InstanceInfo { slot: instance_info.slot + 1, view: 1 };
            let has_proposed = self.already_proposed_slots.contains(&instance_info.slot);
            if self.name == self.leader_elector.get_leader(instance_info.slot + 1, 1) && has_proposed {
                self.ticket = Ticket::new(Some(header.clone()), None, 1, BTreeMap::new()).await;
                if self.enough_coverage(&self.ticket.clone(), self.current_certs.clone()) {
                    let new_prepare_info: PrepareInfo = PrepareInfo { ticket: self.ticket.clone(), proposals: self.current_certs.clone() };
                    let new_info: Info = Info { instance_info: new_instance_info, prepare_info: Some(new_prepare_info), confirm_info: None };
                    self.tx_info
                        .send(new_info)
                        .await
                        .expect("failed to send info to proposer");
                }
            }
        }
    }


    fn process_confirm_info(&mut self, certificate: Certificate, info: ConfirmInfo, instance_info: InstanceInfo) {
        let confirm_valid = self.is_confirm_valid(info.clone());
        if confirm_valid {
            let cert_info = instance_info;
            match self.qcs.get(&cert_info.slot) {
                Some((_, qc_info)) => {
                    if cert_info.view > qc_info.consensus_info.view {
                        self.qcs.insert(info.clone().consensus_info.slot, (certificate.clone(), info.clone()));
                    }
                },
                None => {
                    self.qcs.insert(info.clone().consensus_info.slot, (certificate.clone(), info.clone()));
                }
            }

            if info.clone().cert_type == CertType::Commit {
                self.committed.insert(info.clone().consensus_info.slot, info.clone());
            }
        }
    }

    fn enough_coverage(&mut self, ticket: &Ticket, current_certs: HashMap<PublicKey, Certificate>) -> bool {
        let new_tips: HashMap<&PublicKey, &Certificate> = current_certs.iter()
            .filter(|(pk, cert)| cert.height() > ticket.proposals.get(&pk).unwrap().height())
            .collect();
        
        new_tips.len() as u32 >= self.committee.quorum_threshold()
    }

    //Call process_special_header upon receiving header from data dissemination layer.
    #[async_recursion]
    async fn process_special_header(&mut self, header: Header) -> ConsensusResult<()> {
        //1) Header signature correct => If false, dont need to reply
        header.verify(&self.committee)?;
        header.parent_cert.verify(&self.committee)?;

        self.store_header(&header).await;
        self.store_cert(&header.parent_cert).await;
               
        if header.height() < self.tips.get(&header.origin()).unwrap().height() {
            return Ok(());
        }

        if header.parent_cert.height() > 0 && header.parent_cert.height() < self.current_certs.get(&header.origin()).unwrap().height() {
            return Ok(());
        }

        self.tips.insert(header.origin(), header.clone());
        self.current_certs.insert(header.origin(), header.parent_cert.clone());

        let mut consensus_sigs: Vec<(Info, Signature)> = Vec::new();

        for info in header.info_list.clone() {
            //println!("processing info");
            match info.prepare_info {
                Some(prepare_info) => {
                    self.process_prepare_info(header.clone(), prepare_info.clone(), info.instance_info).await;
                    let sig = self.signature_service.request_signature(info.digest());
                    consensus_sigs.push((info.clone(), sig));
                },
                None => {}
            }

            match info.confirm_info {
                Some(confirm_info) => {
                    self.process_confirm_info(confirm_info.clone(), info.clone(), info.instance_info);
                    let sig = self.signature_service.request_signature(info.digest());
                    consensus_sigs.push((info.clone(), sig));
                },
                None => {}
            }
        }

        self.tx_validation
            .send((header, consensus_sigs))
            .await
            .expect("Failed to send payload");
     
         
        Ok(())
    }

    async fn handle_tc(&mut self, tc: TC) -> ConsensusResult<()> {
        self.process_tc(tc).await
    }

    async fn process_tc(&mut self, tc: TC) -> ConsensusResult<()> {
        Ok(())
    }

    pub async fn run(&mut self) {
        // Upon booting, generate the very first block (if we are the leader).
        // Also, schedule a timer in case we don't hear from the leader.
        self.timer.reset();
        self.tips = Header::genesis_headers(&self.committee);
        self.current_certs = Certificate::genesis_certs(&self.committee);

        // This is the main loop: it processes incoming blocks and votes,
        // and receive timeout notifications from our Timeout Manager.
        loop {
            let result = tokio::select! {
    
                //1) DAG layer headers for special validation
                Some(header) = self.rx_special.recv() => self.process_special_header(header).await,
                
                Some(message) = self.rx_message.recv() => match message {   //Receiving Messages from other Replicas
                    //NO LONGER NEEDED: ConsensusMessage::Propose(block) => self.handle_proposal(&block).await,
                    //NO LONGER NEEDED: ConsensusMessage::Vote(vote) => self.handle_vote(&vote).await,
                    //3) Votes from other replicas to form QC.
                    //5) Timeouts from other replicas to form TC
                    //ConsensusMessage::Timeout(timeout) => self.handle_timeout(timeout).await,
                    //6) Receive TC
                    //ConsensusMessage::TC(tc) => self.handle_tc(tc).await,
                    //7) Sync Headers
                    ConsensusMessage::Header(header) => self.process_special_header(header).await,  //TODO: Sanity check that this is fine. ==> Sync Headers will also cause a vote at the Dag layer to be generated.
                    //8) Sync Headers
                    //ConsensusMessage::Certificate(cert) => self.process_sync_cert(cert).await,  //TODO: Sanity check that this is fine. ==> Sync Headers will also cause a vote at the Dag layer to be generated.
                    _ => panic!("Unexpected protocol message")
                },

                Some(header) = self.rx_loopback_header.recv() => self.process_special_header(header).await,  //Processing Header that we resume via Synchronizer upcall

                //Some((header_dig, ticket)) = self.rx_loopback_process_commit.recv() => self.process_commit(header_dig, ticket).await,
                //Some((header, ticket)) = self.rx_loopback_commit.recv() => self.commit(header, ticket).await,
                
                //NO LONGER NEEDED: Some(block) = self.rx_loopback_header.recv() => self.process_block(&block).await,  //Processing Block that we propose ourselves. (or that we resume via Synchronizer upcall)
                //() = &mut self.timer => self.local_timeout_view().await,
            };
            match result {
                Ok(()) => (),
                Err(ConsensusError::StoreError(e)) => error!("{}", e),
                Err(ConsensusError::SerializationError(e)) => error!("Store corrupted. {}", e),
                Err(e) => warn!("{}", e),
            }
        }
                
    }
}
