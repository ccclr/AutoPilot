use super::*;
use crate::common::{committee_with_base_port, keys};
use config::Parameters;
use crypto::{SecretKey, Hash};
use futures::future::try_join_all;
use network::ReliableSender;
use primary::{Primary, WorkerPrimaryMessage};
use std::collections::{VecDeque, BTreeSet, BTreeMap};
use std::fs;
use std::time::Duration;
use tokio::{sync::mpsc::channel, time::sleep};
use tokio::task::JoinHandle;
use std::net::SocketAddr;
use serial_test::serial;


/*fn spawn_nodes(
    mut keys: Vec<(PublicKey, SecretKey)>,
    committee: Committee,
    store_path: &str,
) -> Vec<JoinHandle<Header>> {
    //Create vector of committer channels.
    let mut committer_channels: VecDeque<(Sender<Certificate>, Receiver<Certificate>)> = keys.iter_mut()
    .map(|_| {
       let (tx_committer, rx_committer) = channel(1);
       (tx_committer, rx_committer)
    })
    .collect();
   let committer_senders: Vec<Sender<Certificate>> = committer_channels.iter_mut()
   .map(|(sender, receiver)| {
       sender.clone()
   })
   .collect();

   //Create vector of consensus channels.
   let mut consensus_channels: VecDeque<(Sender<Certificate>, Receiver<Certificate>)> = keys.iter_mut()
                                .map(|_| {
                                   let (tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
                                   (tx_mempool_to_consensus, rx_mempool_to_consensus)
                                })
                                .collect();
   let consensus_senders: Vec<Sender<Certificate>> = consensus_channels.iter_mut()
                               .map(|(sender, receiver)| {
                                   sender.clone()
                               })
                               .collect();

   let mut stores: VecDeque<Store> = consensus_channels.iter_mut().enumerate().map(|(i, _)| {
       let store_path_1 = format!("{}_{}", store_path, i);
       let _ = fs::remove_dir_all(&store_path_1);
       let store = Store::new(&store_path_1).unwrap();
       store
   }).collect();

   keys.into_iter()
       .enumerate()
       .map(|(i, (name, secret))| {
           let committee = committee.clone();
           let parameters = Parameters {
               timeout_delay: 100,
               ..Parameters::default()
           };
           // let store_path = format!("{}_{}", store_path, i);
           // let _ = fs::remove_dir_all(&store_path);
           // let store = Store::new(&store_path).unwrap();
           let signature_service = SignatureService::new(secret);
           let (tx_consensus_to_mempool, mut rx_consensus_to_mempool) = channel(10);
           //let (_tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
           //let (tx_commit, mut rx_commit) = channel(1);
           //let (tx_committer, rx_committer) = channel(1);

           let (tx_output, mut rx_output) = channel(1);
           let (tx_ticket, mut rx_ticket) = channel(1);
           let (tx_validation, mut rx_validation) = channel(1);
           let (tx_sailfish, rx_sailfish) = channel(1);

           let(tx_pushdown_cert, _rx_pushdown_cert) = channel(1);
           let(tx_request_header_sync, mut rx_request_header_sync) = channel(1);

           let rx_mempool_to_consensus = consensus_channels.pop_front().unwrap().1;
           let rx_committer = committer_channels.pop_front().unwrap().1;

           let committer_sender_channels = committer_senders.clone();
           let consensus_sender_channels = consensus_senders.clone();
           let mut signature_service_copy = signature_service.clone();
           let author = name.clone();
           let mut stores_copy = stores.clone();

           let replica = name.clone();

           let store = stores.pop_front().unwrap();



            // Sink the mempool channel.
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        Some(_) = rx_consensus_to_mempool.recv() => {},
                        Some(_) = rx_request_header_sync.recv() => {},
                        Some(_) = rx_validation.recv() => {},
                          //Receive Ticket
                        Some(tick) = rx_ticket.recv() => {
                            //Ignore all DAG processing TODO: Add a proper primary module here.
                           let ticket: Ticket = tick;
                           
                           //println!("Received a ticket for view {} at replica {}", ticket.view.clone(), author.clone());
                           
                           //Issue Certificate to all nodes. (Note: This is simulating *only* special headers)
                           // Create a new header based on past ticket
                           let parents: BTreeSet<Digest> = BTreeSet::new();
                           let mut payload: Vec<(Digest, u32)> = Vec::new(); // = BTreeMap::new();
                           let new_round = ticket.round.clone() + 1;
                           let new_view = ticket.view.clone() + 1;
                
                           let header: Header = Header::new(author.clone(), new_round, payload.drain(..).collect(), parents, &mut signature_service_copy, true, new_view, None, 0, Some(ticket.clone()), ticket.round.clone(), Some(ticket.digest())).await;
                           // write header to stores
                            for store in &mut stores_copy {
                                let bytes = bincode::serialize(&header).expect("Failed to serialize header");
                                (*store).write(header.id.to_vec(), bytes).await;
                            }
                           
                           
                            //println!("Injecting new Certificate");
                           let cert: Certificate = Certificate { header: header, special_valids: Vec::new(), votes: Vec::new() };

                           //println!("Sending cert to all committers");
                           for sender in &committer_sender_channels{
                                sender.send(cert.clone()).await.expect("failed to send cert to committer");
                           }  
                
                        //println!("Sending cert to all consensus cores");
                           for sender in &consensus_sender_channels{
                               sender.send(cert.clone()).await.expect("failed to send cert to consensus core");
                           }  
                       },
                       
                    }
                }
            });

            // Spawn the consensus engine.
            tokio::spawn(async move {
                Consensus::spawn(
                    name,
                    committee,
                    parameters,
                    signature_service,
                    store,
                    rx_mempool_to_consensus,
                    rx_committer,
                    tx_consensus_to_mempool,
                    tx_output,
                    tx_ticket,
                    tx_validation,
                    rx_sailfish,
                    tx_pushdown_cert,
                    tx_request_header_sync,
                );

                rx_output.recv().await.unwrap()
            })
        })
        .collect()
}

fn spawn_nodes_2(
    mut keys: Vec<(PublicKey, SecretKey)>,
    committee: Committee,
    store_path: &str,
) 
 {

     //Create vector of committer channels.
     let mut committer_channels: VecDeque<(Sender<Certificate>, Receiver<Certificate>)> = keys.iter_mut()
     .map(|_| {
        let (tx_committer, rx_committer) = channel(1);
        (tx_committer, rx_committer)
     })
     .collect();
    let committer_senders: Vec<Sender<Certificate>> = committer_channels.iter_mut()
    .map(|(sender, receiver)| {
        sender.clone()
    })
    .collect();

    //Create vector of consensus channels.
    let mut consensus_channels: VecDeque<(Sender<Certificate>, Receiver<Certificate>)> = keys.iter_mut()
                                 .map(|_| {
                                    let (tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
                                    (tx_mempool_to_consensus, rx_mempool_to_consensus)
                                 })
                                 .collect();
    let consensus_senders: Vec<Sender<Certificate>> = consensus_channels.iter_mut()
                                .map(|(sender, receiver)| {
                                    sender.clone()
                                })
                                .collect();

    let mut stores: VecDeque<Store> = consensus_channels.iter_mut().enumerate().map(|(i, _)| {
        let store_path = format!("{}_{}", store_path, i);
        let _ = fs::remove_dir_all(&store_path);
        let store = Store::new(&store_path).unwrap();
        store
    }).collect();

    let nodes: Vec<i32> = keys.into_iter()
        .enumerate()
        .map(|(i, (name, secret))| {
            let committee = committee.clone();
            let parameters = Parameters {
                timeout_delay: 100,
                ..Parameters::default()
            };
            // let store_path = format!("{}_{}", store_path, i);
            // let _ = fs::remove_dir_all(&store_path);
            // let store = Store::new(&store_path).unwrap();
            let signature_service = SignatureService::new(secret);
            let (tx_consensus_to_mempool, mut rx_consensus_to_mempool) = channel(10);
            //let (_tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
            //let (tx_commit, mut rx_commit) = channel(1);
            //let (tx_committer, rx_committer) = channel(1);

            let (tx_output, mut rx_output) = channel(1);
            let (tx_ticket, mut rx_ticket) = channel(1);
            let (tx_validation, mut rx_validation) = channel(1);
            let (tx_sailfish, rx_sailfish) = channel(1);

            let(tx_pushdown_cert, _rx_pushdown_cert) = channel(1);
            let(tx_request_header_sync, mut rx_request_header_sync) = channel(1);

            let rx_mempool_to_consensus = consensus_channels.pop_front().unwrap().1;
            let rx_committer = committer_channels.pop_front().unwrap().1;

            let committer_sender_channels = committer_senders.clone();
            let consensus_sender_channels = consensus_senders.clone();
            let mut signature_service_copy = signature_service.clone();
            let author = name.clone();
            let mut stores_copy = stores.clone();

            let replica = name.clone();

            let store = stores.pop_front().unwrap();

            // Sink the mempool channel.
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        Some(_) = rx_consensus_to_mempool.recv() => {},
                        Some(_) = rx_request_header_sync.recv() => {},
                        Some(_) = rx_validation.recv() => {},
                          //Receive Ticket
                        Some(tick) = rx_ticket.recv() => {
                            //Ignore all DAG processing TODO: Add a proper primary module here.
                           let ticket: Ticket = tick;
                           
                           //println!("Received a ticket for view {} at replica {}", ticket.view.clone(), author.clone());
                           
                           //Issue Certificate to all nodes. (Note: This is simulating *only* special headers)
                           // Create a new header based on past ticket
                           let parents: BTreeSet<Digest> = BTreeSet::new();
                           let mut payload: Vec<(Digest, u32)> = Vec::new(); // = BTreeMap::new();
                           let new_round = ticket.round.clone() + 1;
                           let new_view = ticket.view.clone() + 1;
                
                           let header: Header = Header::new(author.clone(), new_round, payload.drain(..).collect(), parents, &mut signature_service_copy, true, new_view, None, 0, Some(ticket.clone()), ticket.round.clone(), Some(ticket.digest())).await;
                           // write header to stores
                            for store in &mut stores_copy {
                                let bytes = bincode::serialize(&header).expect("Failed to serialize header");
                                (*store).write(header.id.to_vec(), bytes).await;
                            }
                           
                           
                            //println!("Injecting new Certificate");
                           let cert: Certificate = Certificate { header: header, special_valids: Vec::new(), votes: Vec::new() };

                           //println!("Sending cert to all committers");
                           for sender in &committer_sender_channels{
                                sender.send(cert.clone()).await.expect("failed to send cert to committer");
                           }  
                
                        //println!("Sending cert to all consensus cores");
                           for sender in &consensus_sender_channels{
                               sender.send(cert.clone()).await.expect("failed to send cert to consensus core");
                           }  
                       },
                       
                    }
                }
            });

            // Spawn the consensus engine.
            tokio::spawn(async move {
                Consensus::spawn(
                    name,
                    committee,
                    parameters,
                    signature_service,
                    store,
                    rx_mempool_to_consensus,
                    rx_committer,
                    tx_consensus_to_mempool,
                    tx_output,
                    tx_ticket,
                    tx_validation,
                    rx_sailfish,
                    tx_pushdown_cert,
                    tx_request_header_sync,
                );

                while let Some(header) = rx_output.recv().await {
                    //println!("committed header for round {} at replica {}", header.round, replica.clone());
                    // NOTE: Here goes the application logic.
                }
            });
            0
        })
       .collect();
}


#[tokio::test]
#[serial]
async fn end_to_end_single_header() {
    let committee = committee_with_base_port(15_000);

    // Run all nodes.
    let store_path = ".db_test_end_to_end";
    let handles = spawn_nodes(keys(), committee, store_path);

    //sleep(Duration::from_millis(1000)).await;
    // Ensure all threads terminated correctly.
    let headers = try_join_all(handles).await.unwrap();
    for header in headers.clone() {
        //println!("committed header {:?}", header);
    }
    assert!(headers.windows(2).all(|w| w[0] == w[1]));
}

#[tokio::test]
#[serial]
async fn end_to_end_endless() {
    let committee = committee_with_base_port(15_000);

    // Run all nodes.
    let store_path = ".db_test_end_to_end";
    spawn_nodes_2(keys(), committee, store_path);

    sleep(Duration::from_millis(1000)).await;
   
}


#[tokio::test]
#[serial]
async fn end_to_end_with_dag_single_commit() {
    let committee = committee_with_base_port(23_000);
      // Run all nodes.
      let store_path = ".db_test_end_to_end";
    let handles = spawn_nodes_with_dag_single(keys(), committee, store_path);

    //sleep(Duration::from_millis(1000)).await;
    // Ensure all threads terminated correctly.
    let headers = try_join_all(handles).await.unwrap();
    for header in headers.clone() {
        //println!("committed header {:?}", header);
    }
    assert!(headers.windows(2).all(|w| w[0] == w[1]));
}

fn spawn_nodes_with_dag_single(
    keys: Vec<(PublicKey, SecretKey)>,
    committee: Committee,
    store_path: &str,
) -> Vec<JoinHandle<Header>> {
     keys.into_iter()
        .enumerate()
        .map(|(i, (name, secret))| {
            let committee = committee.clone();
            // FIXME: Need to set timeout at 1 second otherwise timeout messages will be sent
            // Potentially concerning that it takes a long time
            let parameters = Parameters {
                timeout_delay: 1000,
                ..Parameters::default()
            };
            let store_path = format!("{}_{}", store_path, i);
            let _ = fs::remove_dir_all(&store_path);
            let store = Store::new(&store_path).unwrap();
            let signature_service = SignatureService::new(secret);
            let (tx_consensus_to_mempool, mut rx_consensus_to_mempool) = channel(10);
            let (tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
            //let (tx_commit, mut rx_commit) = channel(1);
            let (tx_committer, rx_committer) = channel(1);

            let (tx_output, mut rx_output) = channel(1);
            let (tx_ticket, mut rx_ticket) = channel(1);
            let (tx_validation, mut rx_validation) = channel(1);
            let (tx_sailfish, rx_sailfish) = channel(1);

            let(tx_pushdown_cert, rx_pushdown_cert) = channel(1);
            let(tx_request_header_sync, mut rx_request_header_sync) = channel(1);

            let replica = name.clone();
            //println!("Spawning node {}", replica.clone());

            let name_copy = name.clone();
            let mut store_copy = store.clone();
            let committee_copy = committee.clone();

            let mut network = ReliableSender::new();
    
            // Create a workload inserter.
            tokio::spawn(async move {

                //Create some supply of digests. Simualate Worker Batches
                ////println!("generating workload batches");
                //Note: Should Work without digest supply too: Will just time out

                //TODO: Hack so that digests are automatically written to all replicas? (Avoid any synchronizer)
                // loop {
                //     let worker_id: u32 = 0;
                //     let digest: Digest = Digest::default(); //TODO: Fill this.
    
                //      //store, and send to self
                //     let key = [digest.as_ref(), &worker_id.to_le_bytes()].concat();
                //     store_copy.write(key.to_vec(), Vec::default()).await;
    
                //     let msg = bincode::serialize(&WorkerPrimaryMessage::OurBatch(digest.clone(), worker_id)).expect("failed to serialize batch");
                //     let address = committee_copy.primary(&name_copy).expect("Our public key or worker id is not in the committee").worker_to_primary;
                //     network.send(address, Bytes::from(msg)).await;
                    
    
                //     //send to others
                //     let msg = bincode::serialize(&WorkerPrimaryMessage::OthersBatch(digest, worker_id)).expect("failed to serialize batch");
                //     let addresses : Vec<SocketAddr>= committee_copy.others_primaries(&name_copy).into_iter().map(|(_, x)| x.worker_to_primary).collect();
                //     network.broadcast(addresses, Bytes::from(msg)).await;
                // }
               

            });
            
            
            // Spawn the consensus engine.
            tokio::spawn(async move {

                Primary::spawn(
                    name,
                    committee.clone(),
                    parameters.clone(),
                    signature_service.clone(),
                    store.clone(),
                    /* tx_consensus */ tx_mempool_to_consensus,
                    tx_committer,
                    /* rx_consensus */ rx_consensus_to_mempool,
                    tx_sailfish,
                    rx_ticket,
                    rx_validation,
                    rx_pushdown_cert,
                    rx_request_header_sync,
                );

                Consensus::spawn(
                    name,
                    committee,
                    parameters,
                    signature_service,
                    store,
                    rx_mempool_to_consensus,
                    rx_committer,
                    tx_consensus_to_mempool,
                    tx_output,
                    tx_ticket,
                    tx_validation,
                    rx_sailfish,
                    tx_pushdown_cert,
                    tx_request_header_sync,
                );

                rx_output.recv().await.unwrap()
            })
            
        })
       .collect()
}




#[tokio::test]
#[serial]
async fn end_to_end_with_dag_endless() {
    let committee = committee_with_base_port(23_000);

    // Run all nodes.
    let store_path = ".db_test_end_to_end";
    spawn_nodes_with_dag(keys(), committee, store_path);

    sleep(Duration::from_millis(2000)).await;
   
}

fn spawn_nodes_with_dag(
    keys: Vec<(PublicKey, SecretKey)>,
    committee: Committee,
    store_path: &str,
) 
 {
    let nodes: Vec<i32> = keys.into_iter()
        .enumerate()
        .map(|(i, (name, secret))| {
            let committee = committee.clone();
            let parameters = Parameters {
                timeout_delay: 1000,
                ..Parameters::default()
            };
            let store_path = format!("{}_{}", store_path, i);
            let _ = fs::remove_dir_all(&store_path);
            let store = Store::new(&store_path).unwrap();
            let signature_service = SignatureService::new(secret);
            let (tx_consensus_to_mempool, mut rx_consensus_to_mempool) = channel(10);
            let (tx_mempool_to_consensus, rx_mempool_to_consensus) = channel(1);
            //let (tx_commit, mut rx_commit) = channel(1);
            let (tx_committer, rx_committer) = channel(1);

            let (tx_output, mut rx_output) = channel(1);
            let (tx_ticket, mut rx_ticket) = channel(1);
            let (tx_validation, mut rx_validation) = channel(1);
            let (tx_sailfish, rx_sailfish) = channel(1);

            let(tx_pushdown_cert, rx_pushdown_cert) = channel(1);
            let(tx_request_header_sync, mut rx_request_header_sync) = channel(1);

            let replica = name.clone();
            //println!("Spawning node {}", replica.clone());

            let name_copy = name.clone();
            let mut store_copy = store.clone();
            let committee_copy = committee.clone();

            let mut network = ReliableSender::new();
    
            // Create a workload inserter.
            tokio::spawn(async move {

                //Create some supply of digests. Simualate Worker Batches
                ////println!("generating workload batches");
                //Note: Should Work without digest supply too: Will just time out

                //TODO: Hack so that digests are automatically written to all replicas? (Avoid any synchronizer)
                // loop {
                //     let worker_id: u32 = 0;
                //     let digest: Digest = Digest::default(); //TODO: Fill this.
    
                //      //store, and send to self
                //     let key = [digest.as_ref(), &worker_id.to_le_bytes()].concat();
                //     store_copy.write(key.to_vec(), Vec::default()).await;
    
                //     let msg = bincode::serialize(&WorkerPrimaryMessage::OurBatch(digest.clone(), worker_id)).expect("failed to serialize batch");
                //     let address = committee_copy.primary(&name_copy).expect("Our public key or worker id is not in the committee").worker_to_primary;
                //     network.send(address, Bytes::from(msg)).await;
                    
    
                //     //send to others
                //     let msg = bincode::serialize(&WorkerPrimaryMessage::OthersBatch(digest, worker_id)).expect("failed to serialize batch");
                //     let addresses : Vec<SocketAddr>= committee_copy.others_primaries(&name_copy).into_iter().map(|(_, x)| x.worker_to_primary).collect();
                //     network.broadcast(addresses, Bytes::from(msg)).await;
                // }
               

            });
            
            
            // Spawn the consensus engine.
            tokio::spawn(async move {

                Primary::spawn(
                    name,
                    committee.clone(),
                    parameters.clone(),
                    signature_service.clone(),
                    store.clone(),
                    /* tx_consensus */ tx_mempool_to_consensus,
                    tx_committer,
                    /* rx_consensus */ rx_consensus_to_mempool,
                    tx_sailfish,
                    rx_ticket,
                    rx_validation,
                    rx_pushdown_cert,
                    rx_request_header_sync,
                );

                Consensus::spawn(
                    name,
                    committee,
                    parameters,
                    signature_service,
                    store,
                    rx_mempool_to_consensus,
                    rx_committer,
                    tx_consensus_to_mempool,
                    tx_output,
                    tx_ticket,
                    tx_validation,
                    rx_sailfish,
                    tx_pushdown_cert,
                    tx_request_header_sync,
                );

                while let Some(header) = rx_output.recv().await {
                    if header.is_special {
                        //println!("SPECIAL committed view {}, view_round {} at replica {}", header.view, header.round, replica.clone());
                    }
                    else{
                        //println!("NORMAL: committed header for round {} at replica {}", header.round, replica.clone());
                    }
                    
                    // NOTE: Here goes the application logic.
                }
            });
            0
        })
       .collect();
}*/
