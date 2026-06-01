use crate::committer::Committer;
use crate::core::Core;
//use crate::error::ConsensusError;
use primary::{error::{ConsensusError}, messages::{PrepareInfo, ConfirmInfo}};
use crate::helper::Helper;
use crate::leader::LeaderElector;
use crate::mempool::MempoolDriver;
use primary::messages::{Header, Certificate, Timeout, TC, Ticket, Vote};
//use crate::proposer::Proposer;
use crate::synchronizer::Synchronizer;
use async_trait::async_trait;
use bytes::Bytes;
use config::{Committee, Parameters};
use crypto::{Digest, PublicKey, SignatureService};
use futures::SinkExt as _;
use log::info;
use network::{MessageHandler, Receiver as NetworkReceiver, Writer};
use serde::{Deserialize, Serialize};
use std::error::Error;
use store::Store;
use tokio::sync::mpsc::{channel, Receiver, Sender};

#[cfg(test)]
#[path = "tests/consensus_tests.rs"]
pub mod consensus_tests;

/// The default channel capacity for each channel of the consensus.
pub const CHANNEL_CAPACITY: usize = 1_000;

/// The consensus view number.
pub type View= u64;
/// The Dag round number.
pub type Round= u64;
// The consensus slot (sequence) number
pub type Slot= u64;

#[derive(Serialize, Deserialize, Debug)]
pub enum ConsensusMessage {
    Vote(Vote),     //No longer used
    Timeout(Timeout),
    SyncRequest(Digest, PublicKey), //Note: These Digests are now for Headers
    SyncRequestCert(Digest, PublicKey),
    Header(Header),
    Certificate(Certificate),
    CommitHeader(Header), //reception of this header should call commit with a dummy ticket. (It does not need to be processed since it's transitively validated -- 
                            // if we wanted to, we'd have to start a waiter that waits for the header to be stored; and pass the received header to Dag process header)
                            // We DO need to do this, because we want to call the missing_payload synchronizer...
}

pub struct Consensus;

impl Consensus {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        name: PublicKey,
        committee: Committee,
        parameters: Parameters,
        signature_service: SignatureService,
        store: Store,
        rx_consensus: Receiver<Certificate>,    // This is the channel used to upcall special certs from the DAG
        rx_committer: Receiver<Certificate>,  // This is the channel used to send all certs to committer (previously this was called mempool -- maybe rename for clarity)
        tx_mempool: Sender<Certificate>,
        tx_output: Sender<Header>,
        tx_ticket: Sender<PrepareInfo>,//Sender<(View, Round, Ticket)>,
        tx_validation: Sender<(Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>)>,
        rx_sailfish: Receiver<Header>,
        tx_pushdown_cert: Sender<Certificate>,
        tx_request_header_sync: Sender<Digest>,
    ) {
        // NOTE: This log entry is used to compute performance.
        parameters.log();

        let (tx_message, rx_message) = channel(CHANNEL_CAPACITY);    //This is the messaging channel for consensus messages
        //let (tx_proposer, _rx_proposer) = channel(CHANNEL_CAPACITY);   //TODO: Remove, no longer used
        let (tx_helper_header, rx_helper_header) = channel(CHANNEL_CAPACITY);
        let (tx_helper_cert, rx_helper_cert) = channel(CHANNEL_CAPACITY);
        let (tx_commit, rx_commit) = channel(CHANNEL_CAPACITY);
        //let (tx_mempool_copy, rx_mempool_copy) = channel(CHANNEL_CAPACITY);
        

        let (tx_loop_header, rx_loop_header) = channel(CHANNEL_CAPACITY);
        let (tx_loop_cert, rx_loop_cert) = channel(CHANNEL_CAPACITY);

        let (tx_loopback_process_commit, rx_loopback_process_commit) = channel(CHANNEL_CAPACITY);
        let (tx_loopback_commit, rx_loopback_commit) = channel(CHANNEL_CAPACITY);

        //let (tx_sailfish, rx_sailfish) = channel(CHANNEL_CAPACITY);
        //let (tx_dag, rx_dag) = channel(CHANNEL_CAPACITY);

        // Spawn the network receiver.
        let mut address = committee
            .consensus(&name)
            .expect("Our public key is not in the committee")
            .consensus_to_consensus;
        address.set_ip("0.0.0.0".parse().unwrap());
        NetworkReceiver::spawn(
            address,
            /* handler */
            ConsensusReceiverHandler {
                tx_message,
                tx_helper_header,
                tx_helper_cert
            },
        );
        info!(
            "Node {} listening to consensus messages on {}",
            name, address
        );

        // Make the leader election module.
        let leader_elector = LeaderElector::new(committee.clone());

        // Make the mempool driver.
        let mempool_driver = MempoolDriver::new(committee.clone(), tx_mempool);

        // Make the synchronizer.
        let synchronizer = Synchronizer::new(
            name,
            committee.clone(),
            store.clone(),
            tx_loop_header.clone(),
            tx_loop_cert,
            tx_request_header_sync,
            tx_loopback_process_commit,
            tx_loopback_commit,
            parameters.sync_retry_delay,
        );

        // Spawn the consensus core.
        Core::spawn(
            name,
            committee.clone(),
            signature_service.clone(),
            store.clone(),
            leader_elector,
            mempool_driver,
            synchronizer,
            parameters.timeout_delay,
            /* rx_message */ rx_message,
            /* rx_consensus */ rx_consensus,
            rx_loop_header,
            rx_loop_cert,
            tx_pushdown_cert,
            //tx_proposer,
            tx_commit,
            tx_validation,
            tx_ticket,
            /*rx_special */ rx_sailfish,
            rx_loopback_process_commit,
            rx_loopback_commit,
        );

        // Commits the mempool certificates and their sub-dag.
        Committer::spawn(
            committee.clone(),
            store.clone(),
            parameters.gc_depth,
            /* rx_mempool */ rx_committer, //Receive certs directly.
            rx_commit,
            tx_output,
        );

        // Spawn the block proposer.
        // Proposer::spawn(
        //     name,
        //     committee.clone(),
        //     signature_service,
        //     /* rx_consensus */ rx_sailfish,
        //     rx_mempool,
        //     /* rx_message */ rx_proposer,
        //     tx_loop_header,
        //     tx_mempool_copy,
        // );

        // Spawn the helper module.
        Helper::spawn(committee, store, /* rx_requests */ rx_helper_header, rx_helper_cert);
    }
}

/// Defines how the network receiver handles incoming primary messages.
#[derive(Clone)]
struct ConsensusReceiverHandler {
    tx_message: Sender<ConsensusMessage>,
    tx_helper_header: Sender<(Digest, PublicKey)>,
    tx_helper_cert: Sender<(Digest, PublicKey)>,
}

#[async_trait]
impl MessageHandler for ConsensusReceiverHandler {
    async fn dispatch(&self, writer: &mut Writer, serialized: Bytes) -> Result<(), Box<dyn Error>> {
        // Deserialize and parse the message.
        match bincode::deserialize(&serialized).map_err(ConsensusError::SerializationError)? {
            ConsensusMessage::SyncRequest(missing, origin) => self
                .tx_helper_header
                .send((missing, origin))
                .await
                .expect("Failed to send consensus message"),
            ConsensusMessage::SyncRequestCert(missing, origin) => self
                .tx_helper_cert
                .send((missing, origin))
                .await
                .expect("Failed to send consensus message"),
            /*message @ ConsensusMessage::Propose(..) => {
                // Reply with an ACK.
                let _ = writer.send(Bytes::from("Ack")).await;

                // Pass the message to the consensus core.
                self.tx_message
                    .send(message)
                    .await
                    .expect("Failed to send consensus message")
            }*/
            message => self
                .tx_message
                .send(message)
                .await
                .expect("Failed to send consensus message"),
        }
        Ok(())
    }
}
