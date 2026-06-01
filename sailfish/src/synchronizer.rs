use crate::consensus::{ConsensusMessage, CHANNEL_CAPACITY};
use primary::error::{ConsensusResult, ConsensusError};
use primary::messages::{Header, Certificate, Ticket};
use bytes::Bytes;
use config::Committee;
use crypto::Hash as _;
use crypto::{Digest, PublicKey};

use futures::stream::futures_unordered::FuturesUnordered;
use futures::stream::StreamExt as _;
use log::{debug, error};
use network::SimpleSender;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{SystemTime, UNIX_EPOCH};
use store::Store;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::{sleep, Duration, Instant};

// use ed25519_dalek::Digest as _;
// use ed25519_dalek::Sha512;
// use std::convert::TryInto;

#[cfg(test)]
#[path = "tests/synchronizer_tests.rs"]
pub mod synchronizer_tests;

const TIMER_ACCURACY: u64 = 5_000;

pub struct Synchronizer {
    store: Store,
    committee: Committee,
    inner_channel: Sender<(Header, Digest)>,
    inner_channel_cert: Sender<Certificate>,

    inner_channel_header: Sender<(Digest, Ticket)>,
    inner_channel_ticket: Sender<(Header, Ticket)>,
    inner_channel_prev_special: Sender<(Header, Digest)>,
    //tx_dag_request_header_sync: Sender<Digest>,
    tx_loopback_commit: Sender<(Header, Ticket)>,
}

impl Synchronizer {

    async fn commit_header_waiter(mut store: Store, wait_on: Digest, ticket: Ticket) -> ConsensusResult<(Digest, Ticket)> {
        let _ = store.notify_read(wait_on.to_vec()).await?;
        Ok((wait_on, ticket))
    }

    async fn ticket_commit_waiter(mut store: Store, wait_on: Digest, header: Header, ticket: Ticket) -> ConsensusResult<(Header, Ticket)> {
        let _ = store.notify_read(wait_on.to_vec()).await?;
        Ok((header, ticket))
    }

    async fn prev_special_waiter(mut store: Store, wait_on: Digest, deliver: Header) -> ConsensusResult<(Header, Digest)> {
        let _ = store.notify_read(wait_on.to_vec()).await?;
        Ok((deliver, wait_on))
    }

    async fn header_waiter(mut store: Store, wait_on: Digest, deliver: Header) -> ConsensusResult<(Header, Digest)> {
        let _ = store.notify_read(wait_on.to_vec()).await?;
        Ok((deliver, wait_on))
    }

    async fn cert_waiter(mut store: Store, wait_on: Digest, deliver: Certificate) -> ConsensusResult<(Certificate, Digest)> {
        let _ = store.notify_read(wait_on.to_vec()).await?;
        Ok((deliver, wait_on))
    }



    pub fn new(
        name: PublicKey,
        committee: Committee,
        store: Store,
        tx_loopback_header: Sender<Header>,
        tx_loopback_certs: Sender<Certificate>,

        tx_request_header_sync: Sender<Digest>,
        tx_loopback_process_commit: Sender<(Digest, Ticket)>,
        tx_loopback_commit: Sender<(Header, Ticket)>,

        sync_retry_delay: u64,
    ) -> Self {
        let mut network = SimpleSender::new();
        let (tx_inner, mut rx_inner): (_, Receiver<(Header, Digest)>) = channel(CHANNEL_CAPACITY);
        let (tx_cert, mut rx_cert): (_, Receiver<Certificate>) = channel(CHANNEL_CAPACITY);
        let (tx_header, mut rx_header): (_, Receiver<(Digest, Ticket)>) = channel(CHANNEL_CAPACITY);
        let (tx_ticket, mut rx_ticket): (_, Receiver<(Header, Ticket)>) = channel(CHANNEL_CAPACITY);
        let (tx_prev_special, mut rx_prev_special): (_, Receiver<(Header, Digest)>) = channel(CHANNEL_CAPACITY);


        let tx_loopback_commit_copy = tx_loopback_commit.clone();
        let store_copy = store.clone();
        let committee_copy = committee.clone();
        tokio::spawn(async move {

            let mut waiting_headers = FuturesUnordered::new();
            /*let mut waiting_certs = FuturesUnordered::new();
            let mut waiting_tickets = FuturesUnordered::new();
            let mut waiting_prev_special = FuturesUnordered::new();*/
            let mut waiting_commit_headers = FuturesUnordered::new();

            let mut pending = HashSet::new();
            let mut requests = HashMap::new();
            //let mut cert_requests = HashMap::new();

            let timer = sleep(Duration::from_millis(TIMER_ACCURACY));
            tokio::pin!(timer);
            loop {
                tokio::select! {
                    Some((header_digest, ticket)) = rx_header.recv() => {
                        //TODO: add duplicate checks.
                         //Create waiter for header to be stored. Upon trigger, call Commit_wrapper()
                        let fut = Self::commit_header_waiter(store_copy.clone(), header_digest.clone(), ticket.clone());
                        waiting_commit_headers.push(fut);
                        
                        //downcall to DAG, and let it synchronize.
                        tx_request_header_sync.send(header_digest).await.expect("requesting header sync failed"); 
                         //Altneratively: Send out a CommitHeader Request -> upon reception, call down to Dag to process_header. (ensures payload is there)

                    },
                    //header has been added to store => can re-trigger commit.
                    Some(result) = waiting_commit_headers.next() => match result {
                        Ok((header_digest, ticket)) => {
                            if let Err(e) = tx_loopback_process_commit.send((header_digest, ticket)).await { // loopback to core. ==> receiver calls process_commit, which fetches header and calls commit
                                panic!("Failed to send message through core commit_header channel: {}", e);
                            }
                        },
                        Err(e) => error!("{}", e)
                    },

                   
                    Some((header, ticket)) = rx_ticket.recv() => {
                        /*let consensus_parent = header.consensus_parent.clone().unwrap();
                        //start waiter.
                        let fut = Self::ticket_commit_waiter(store_copy.clone(), consensus_parent, header.clone(), ticket.clone());
                        waiting_tickets.push(fut);*/

                    },
                     //parent ticket has been added to store => re-trigger commit with buffered header/ticket
                    /*Some(result) = waiting_tickets.next() => match result {
                        Ok((header, ticket)) => {
                            debug!("parent ticket has committed. Waking up!");
                            if let Err(e) = tx_loopback_commit.send((header, ticket)).await { //loopback to core. ==> receiver calls commit(header,ticket)
                                panic!("Failed to send message through core commit_header channel: {}", e);
                            }
                        },
                        Err(e) => error!("{}", e)
                    },*/

                    Some((header, ticket)) = rx_prev_special.recv() => {
                        //let consensus_parent = header.consensus_parent.clone().unwrap();
                        //start waiter.
                        //let fut = Self::prev_special_waiter(store_copy.clone(), header.clone(), ticket.clone());
                        //waiting_prev_special.push(fut);
                    },


                    ////////////////////// UNUSED/DEPRECATED below
                    /* */
                    
                    Some((header, parent_dig)) = rx_inner.recv() => {
                        if pending.insert(header.digest()) {
                            let parent = parent_dig.clone();
                            let author = header.author;
                            let fut = Self::header_waiter(store_copy.clone(), parent.clone(), header);
                            waiting_headers.push(fut);

                            if !requests.contains_key(&parent){
                                debug!("Requesting sync for header {}", parent);
                                let now = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Failed to measure time")
                                    .as_millis();
                                requests.insert(parent.clone(), now);
                                let address = committee
                                    .consensus(&author)
                                    .expect("Author of valid header is not in the committee")
                                    .consensus_to_consensus;
                                let message = ConsensusMessage::SyncRequest(parent, name);
                                let message = bincode::serialize(&message)
                                    .expect("Failed to serialize sync request");
                                network.send(address, Bytes::from(message)).await;
                            }
                        }
                    },

                    Some(result) = waiting_headers.next() => match result {
                        Ok((header, parent)) => {
                            let _ = pending.remove(&header.digest());
                            let _ = requests.remove(&parent);
                            if let Err(e) = tx_loopback_header.send(header).await {
                                panic!("Failed to send message through core channel: {}", e);
                            }
                        },
                        Err(e) => error!("{}", e)
                    },

                    Some(cert) = rx_cert.recv() => {
                        /*if pending.insert(cert.digest()) {
                            let parent_cert_digest = cert.header.consensus_parent.clone().unwrap();
                            let fut = Self::cert_waiter(store_copy.clone(), parent_cert_digest.clone(), cert.clone());
                            waiting_certs.push(fut);

                            if !requests.contains_key(&parent_cert_digest){    
                                debug!("Requesting sync for certificate {}", parent_cert_digest);
                                let now = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Failed to measure time")
                                    .as_millis();
                                requests.insert(parent_cert_digest.clone(), now);
                                let address = committee
                                    .consensus(&cert.header.author)
                                    .expect("Author of valid header is not in the committee")
                                    .consensus_to_consensus;
                                let message = ConsensusMessage::SyncRequestCert(parent_cert_digest, name);
                                let message = bincode::serialize(&message)
                                    .expect("Failed to serialize sync request");
                                network.send(address, Bytes::from(message)).await;
                            }
                        }*/
                    },
                    /*Some(result) = waiting_certs.next() => match result {
                        Ok((cert, parent)) => {
                            let _ = pending.remove(&cert.digest());
                            let _ = requests.remove(&parent);
                            if let Err(e) = tx_loopback_certs.send(cert).await {
                                panic!("Failed to send message through core channel: {}", e);
                            }
                        },
                        Err(e) => error!("{}", e)
                    },*/
                  
                    // Some(block) = rx_inner.recv() => {
                    //     if pending.insert(block.digest()) {
                    //         let parent = block.parent().clone();
                    //         let author = block.author;
                    //         let fut = Self::waiter(store_copy.clone(), parent.clone(), block);
                    //         waiting.push(fut);

                    //         if !requests.contains_key(&parent){
                    //             debug!("Requesting sync for block {}", parent);
                    //             let now = SystemTime::now()
                    //                 .duration_since(UNIX_EPOCH)
                    //                 .expect("Failed to measure time")
                    //                 .as_millis();
                    //             requests.insert(parent.clone(), now);
                    //             let address = committee
                    //                 .consensus(&author)
                    //                 .expect("Author of valid block is not in the committee")
                    //                 .consensus_to_consensus;
                    //             let message = ConsensusMessage::SyncRequest(parent, name);
                    //             let message = bincode::serialize(&message)
                    //                 .expect("Failed to serialize sync request");
                    //             network.send(address, Bytes::from(message)).await;
                    //         }
                    //     }
                    // },
                   
                    () = &mut timer => {
                        // This implements the 'perfect point to point link' abstraction.
                        for (digest, timestamp) in &requests {
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Failed to measure time")
                                .as_millis();
                            if timestamp + (sync_retry_delay as u128) < now {
                                debug!("Requesting sync for block {} (retry)", digest);
                                let addresses = committee
                                    .others_consensus(&name)
                                    .into_iter()
                                    .map(|(_, x)| x.consensus_to_consensus)
                                    .collect();
                                let message = ConsensusMessage::SyncRequest(digest.clone(), name);
                                let message = bincode::serialize(&message)
                                    .expect("Failed to serialize sync request");
                                network.broadcast(addresses, Bytes::from(message)).await;
                            }
                        }
                        timer.as_mut().reset(Instant::now() + Duration::from_millis(TIMER_ACCURACY));
                    }
                }
            }
        });
        Self {
            store,
            committee: committee_copy,
            inner_channel: tx_inner,
            inner_channel_cert: tx_cert,
            inner_channel_header: tx_header,
            inner_channel_ticket: tx_ticket,
            tx_loopback_commit: tx_loopback_commit_copy,
            inner_channel_prev_special: tx_prev_special,
            //tx_dag_request_header_sync: tx_request_header_sync,
        }
    }

    
    // pub async fn get_parent_block(&mut self, block: &Block) -> ConsensusResult<Option<Block>> {
    //     if block.qc == QC::genesis() {
    //         return Ok(Some(Block::genesis()));
    //     }
    //     let parent = block.parent();
    //     match self.store.read(parent.to_vec()).await? {
    //         Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
    //         None => {
    //             if let Err(e) = self.inner_channel.send(block.clone()).await {
    //                 panic!("Failed to send request to synchronizer: {}", e);
    //             }
    //             Ok(None)
    //         }
    //     }
    // }

   
    pub async fn get_commit_header(&mut self, header_digest: Digest, new_ticket: &Ticket) -> ConsensusResult<Option<Header>>{
        
        //read digest
        debug!("getting header");
        match self.store.read(header_digest.to_vec()).await? {
             //If we have header ==> reply
            Some(bytes) => return Ok(Some(bincode::deserialize(&bytes)?)),
            None => {
                debug!("don't have header");
                //Else: 
            //Send to synchronizer loop via channel: 
                if let Err(e) = self.inner_channel_header.send((header_digest.clone(), new_ticket.clone())).await {
                    panic!("Failed to send header request to synchronizer: {}", e)
                }
            }
        }
        Ok(None)
    }

    pub async fn deliver_parent_ticket(&mut self, header: &Header, new_ticket: &Ticket) -> ConsensusResult<bool>{
       
        /*let parent: Digest = header.consensus_parent.clone().unwrap();

        //if ticket == genesis, return true. ==> Already have genesis, don't need to commit it
        if parent == Ticket::genesis(&self.committee).digest(){
            return Ok(true);
        }
      
        //read store for parent ticket digest
        //If we have => return true
        if self.store.read(parent.to_vec()).await?.is_some(){
            return Ok(true);
        }*/

        debug!("parent ticket not in store");
        //Else: return false =>
        //AND
        /*let parent_ticket: Ticket = header.ticket.clone().unwrap();
        let parent_ticket_header_dig: Digest = parent_ticket.hash.clone();
            //Send to synchronizer loop via channel: 
            //start a waiter that restarts header and waits for ticket_digest
        if let Err(e) = self.inner_channel_ticket.send((header.clone(), new_ticket.clone())).await {
             panic!("Failed to send header request to synchronizer: {}", e)
        }

        //check if we have the ticket header stored. 

        match self.get_commit_header(parent_ticket_header_dig, &parent_ticket).await? {
            None => {}, //If no: get_commit_header will start sync, and eventually call process_commit(parent_ticket_header_dig, parent_ticket)

            Some(parent_ticket_header) => {  //If yes: issue commit(parent_ticket_header, parent_ticket)
                debug!("parent ticket header in store! starting commit for parent");
               if let Err(e) = self.tx_loopback_commit.send((parent_ticket_header, parent_ticket)).await{
                panic!("Failed to loopback commit to core {}", e);
               }
            }
        }*/
       
        Ok(false)
    }

    pub async fn get_prev_special_header(&mut self, header: &Header, ticket: Digest) -> ConsensusResult<Option<Header>>{

        if ticket == Ticket::genesis(&self.committee).digest(){
            //return Ok(Header::genesis(&self.committee));
            return Ok(None);
        }

        match self.store.read(ticket.to_vec()).await? {
            None => {
                panic!{"Should have previous special header by now"}
                return Err(ConsensusError::UncommittedParentTicket);
            }
            Some(bytes) => {
                let prev_special_header: Header = bincode::deserialize(&bytes)?;

                /*match self.store.read(ticket.digest().to_vec()).await? {
                    None => {
                        panic!{"Should have header of the parent ticket by now"}
                        return Err(ConsensusError::MissingParentTicketHeader);
                    }
                    Some(bytes) => {
                        Ok(Some(bincode::deserialize(&bytes)?))
                    }
                }*/ 
                Ok(Some(prev_special_header))
            }
        }
    }

    pub async fn get_proposals(&mut self, header: &Header) -> ConsensusResult<BTreeMap<PublicKey, Certificate>> {
        /*if header.consensus_info.is_none() {
            return Ok(BTreeMap::new());
        }
        
        let proposal_digests: BTreeMap<PublicKey, Digest> = header.consensus_info.unwrap().proposals;

        let mut missing: Vec<Digest> = Vec::new();
        let mut proposals: BTreeMap<PublicKey, Certificate> = BTreeMap::new();

        for (pub_key, digest) in proposal_digests.iter() {
            if let Some(genesis) = Certificate::genesis(&self.committee)
                .iter()
                .find(|(x, _)| x == digest)
                .map(|(_, x)| x)
            {
                proposals.insert(pub_key, genesis.clone());
                continue;
            }

            match self.store.read(digest.to_vec()).await? {
                Some(certificate) => proposals.insert(pub_key, bincode::deserialize(&certificate)?),
                None => missing.push(digest.clone()),
            };
        }

        if missing.is_empty() {
            return Ok(proposals);
        }*/

        Ok(BTreeMap::new())

         //if ticket == genesis, return empty ==> Don't need to commit genesis
        /*if parent == Ticket::genesis(&self.committee).digest(){
            //return Ok(Header::genesis(&self.committee));
            return Ok(None);
        }

        match self.store.read(parent.to_vec()).await? {
            None => {
                panic!{"Should have parent ticket by now"}
                return Err(ConsensusError::UncommittedParentTicket);
            }
            Some(bytes) => {
                let ticket:Ticket = bincode::deserialize(&bytes)?;

                match self.store.read(ticket.digest().to_vec()).await? {
                    None => {
                        panic!{"Should have header of the parent ticket by now"}
                        return Err(ConsensusError::MissingParentTicketHeader);
                    }
                    Some(bytes) => {
                        Ok(Some(bincode::deserialize(&bytes)?))
                    }
                }

            }
        }*/
    }




    //
    pub async fn get_parent_header(&mut self, header: &Header) -> ConsensusResult<Option<Header>> {
        
        /*let parent: Digest = header.consensus_parent.clone().unwrap();

         //if ticket == genesis, return empty ==> Don't need to commit genesis
        if parent == Ticket::genesis(&self.committee).digest(){
            //return Ok(Header::genesis(&self.committee));
            return Ok(None);
        }

        match self.store.read(parent.to_vec()).await? {
            None => {
                panic!{"Should have parent ticket by now"}
                return Err(ConsensusError::UncommittedParentTicket);
            }
            Some(bytes) => {
                let ticket:Ticket = bincode::deserialize(&bytes)?;

                match self.store.read(ticket.digest().to_vec()).await? {
                    None => {
                        panic!{"Should have header of the parent ticket by now"}
                        return Err(ConsensusError::MissingParentTicketHeader);
                    }
                    Some(bytes) => {
                        Ok(Some(bincode::deserialize(&bytes)?))
                    }
                }

            }
        }*/
        Ok(None)
    }


    /////////////////////////////////// UNUSED/DEPRECATED BELOW
    /*

    pub async fn get_parent_cert(&mut self, cert: &Certificate) -> ConsensusResult<Option<Certificate>> {
        
        match self.store.read(cert.header.consensus_parent.as_ref().unwrap().to_vec()).await? {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => {
                Ok(None)
            }
        }
    }


    pub async fn get_special_parent_header(&mut self, header: &Header) -> ConsensusResult<Option<Header>> {
        let parent = header.special_parent.clone();
        let mut return_value = Ok(None);

        if parent.is_some() {
            match self.store.read(parent.clone().unwrap().to_vec()).await? {
                Some(bytes) => return_value = Ok(Some(bincode::deserialize(&bytes)?)),
                None  => {
                    if let Err(e) = self.inner_channel.send((header.clone(), parent.unwrap())).await {
                        panic!("Failed to send request to synchronizer: {}", e);
                    }
                }
            }
        }
        return_value
    }

    pub async fn get_cert(&mut self, header: &Header) -> ConsensusResult<Option<Certificate>> {

        // let cert_dig: Digest = Certificate {
        //     header: header.clone(),
        //     ..Certificate::default()
        // }.digest();

         //directly generating the hash avoids copying the header.
        let mut hasher = Sha512::new();
        hasher.update(&header.id); //== parent_header.id
        hasher.update(&header.round().to_le_bytes()); 
        hasher.update(&header.origin()); //parent_header.origin = child_header_origin
        let cert_digest = Digest(hasher.finalize().as_slice()[..32].try_into().unwrap());


        match self.store.read(cert_digest.to_vec()).await? {   
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None  => {
                panic!("Ancestor cert not available");
                Ok(None)
            }
        }
    }


    // pub async fn get_ancestors(
    //     &mut self,
    //     block: &Block,
    // ) -> ConsensusResult<Option<(Block, Block)>> {
    //     let b1 = match self.get_parent_block(block).await? {
    //         Some(b) => b,
    //         None => return Ok(None),
    //     };
    //     let b0 = self
    //         .get_parent_block(&b1)
    //         .await?
    //         .expect("We should have all ancestors of delivered blocks");
    //     Ok(Some((b0, b1)))
    // }

    pub async fn deliver_consensus_ancestor(&mut self, cert: &Certificate) -> ConsensusResult<bool>{

        if cert.header.special_parent_round == 0 {
            return Ok(true);
        }

        if self.store.read(cert.header.consensus_parent.as_ref().unwrap().to_vec()).await?.is_none(){
            if let Err(e) = self.inner_channel_cert.send(cert.clone()).await {
                panic!("Failed to send request to synchronizer: {}", e);
           }
           return Ok(false);
        }

        Ok(true)
    }

    pub async fn deliver_self(&mut self, cert: &Certificate) -> ConsensusResult<bool>{

        if self.store.read(cert.header.id.to_vec()).await?.is_none(){
        //     if let Err(e) = self.inner_channel_cert.send(cert.clone()).await {
        //         panic!("Failed to send request to synchronizer: {}", e);
        //    }
           return Ok(false);
        }

        Ok(true)
    }

    */
}
