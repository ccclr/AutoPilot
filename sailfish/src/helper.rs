use crate::consensus::ConsensusMessage;
use bytes::Bytes;
use config::Committee;
use crypto::{Digest, PublicKey};
use log::warn;
use network::SimpleSender;
use primary::{Header, Certificate};
use store::Store;
use tokio::sync::mpsc::Receiver;
//use std::any::{self, Any};

#[cfg(test)]
#[path = "tests/helper_tests.rs"]
pub mod helper_tests;

/// A task dedicated to help other authorities by replying to their sync requests.
pub struct Helper {
    /// The committee information.
    committee: Committee,
    /// The persistent storage.
    store: Store,
    /// Input channel to receive sync requests.
    rx_requests_header: Receiver<(Digest, PublicKey)>,
    rx_requests_cert: Receiver<(Digest, PublicKey)>,
    /// A network sender to reply to the sync requests.
    network: SimpleSender,
}

impl Helper {
    pub fn spawn(committee: Committee, store: Store, rx_requests_header: Receiver<(Digest, PublicKey)>, rx_requests_cert: Receiver<(Digest, PublicKey)>) {
        tokio::spawn(async move {
            Self {
                committee,
                store,
                rx_requests_header,
                rx_requests_cert,
                network: SimpleSender::new(),
            }
            .run()
            .await;
        });
    }

    async fn run(&mut self) {
        loop{
            tokio::select! {
                Some((digest, origin)) = self.rx_requests_header.recv() => {
                        // TODO [issue #58]: Do some accounting to prevent bad nodes from monopolizing our resources.
            
                        // get the requestors address.
                        let address = match self.committee.consensus(&origin) {
                            Ok(x) => x.consensus_to_consensus,
                            Err(e) => {
                                warn!("Received unexpected sync request: {}", e);
                                continue;
                            }
                        };
            
                        // Reply to the request (if we can).
                        if let Some(bytes) = self
                            .store
                            .read(digest.to_vec())
                            .await
                            .expect("Failed to read from storage")
                        {
                            let header: Header=
                                bincode::deserialize(&bytes).expect("Failed to deserialize our own header");
                            let message = bincode::serialize(&ConsensusMessage::Header(header)) //TODO: Change this message.
                                .expect("Failed to serialize block");
                            self.network.send(address, Bytes::from(message)).await;
                        }
                },
                Some((digest, origin)) = self.rx_requests_cert.recv() => {
                    // TODO [issue #58]: Do some accounting to prevent bad nodes from monopolizing our resources.
        
                    // get the requestors address.
                    let address = match self.committee.consensus(&origin) {
                        Ok(x) => x.consensus_to_consensus,
                        Err(e) => {
                            warn!("Received unexpected sync request: {}", e);
                            continue;
                        }
                    };
        
                    // Reply to the request (if we can).
                    if let Some(bytes) = self
                        .store
                        .read(digest.to_vec())
                        .await
                        .expect("Failed to read from storage")
                    {
                        let cert: Certificate=
                            bincode::deserialize(&bytes).expect("Failed to deserialize our own header");
                        let message = bincode::serialize(&ConsensusMessage::Certificate(cert)) //TODO: Change this message.
                            .expect("Failed to serialize block");
                        self.network.send(address, Bytes::from(message)).await;
                    }
        },

        }
        // while let Some((digest, origin)) = self.rx_requests.recv().await {
        //     // TODO [issue #58]: Do some accounting to prevent bad nodes from monopolizing our resources.

        //     // get the requestors address.
        //     let address = match self.committee.consensus(&origin) {
        //         Ok(x) => x.consensus_to_consensus,
        //         Err(e) => {
        //             warn!("Received unexpected sync request: {}", e);
        //             continue;
        //         }
        //     };

        //     // Reply to the request (if we can).
        //     if let Some(bytes) = self
        //         .store
        //         .read(digest.to_vec())
        //         .await
        //         .expect("Failed to read from storage")
        //     {
        //         let header=
        //             bincode::deserialize(&bytes).expect("Failed to deserialize our own header");
        //         let message = bincode::serialize(&ConsensusMessage::Header(header)) //TODO: Change this message.
        //             .expect("Failed to serialize block");
        //         self.network.send(address, Bytes::from(message)).await;
        //     }
        // }
        }   
    }
}

//  if let Some(bytes) = self
// .store
// .read(digest.to_vec())
// .await
// .expect("Failed to read from storage")
// {
// let any_value = bincode::deserialize(&bytes).expect("Failed to deserialize our own header") as &dyn Any;
// if any_value.is::<Header>() {
//     match any_value.downcast_mut::<Header>() {
//         Some(header) => {
//             let message = bincode::serialize(&ConsensusMessage::Header(*header)) //TODO: Change this message.
//             .expect("Failed to serialize block");
//              self.network.send(address, Bytes::from(message)).await;
//         }
//         None => {}
//     }
 
   
// }
// else if any_value.is::<Certificate>() {
//     match any_value.downcast_mut::<Certificate>() {
//         Some(cert) => {
//             let message = bincode::serialize(&ConsensusMessage::Certificate(*cert)) //TODO: Change this message.
//             .expect("Failed to serialize block");
//              self.network.send(address, Bytes::from(message)).await;
//         }
//         None => {}
//     }
// }
// else{
//     panic!("cannot decode");
// }