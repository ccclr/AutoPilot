use super::*;
use crate::common::{committee, committee_with_base_port, keys, listener};
use crypto::{SecretKey, Signature};
use primary::messages::{Header, TC, Vote};
use futures::future::try_join_all;
use std::{fs, collections::BTreeMap, collections::BTreeSet, time::Duration, vec};
use tokio::{sync::mpsc::channel, time::sleep};
use serial_test::serial;

fn core(
    name: PublicKey,
    secret: SecretKey,
    committee: Committee,
    store_path: &str,
) -> (
    Sender<ConsensusMessage>,
    //Receiver<ProposerMessage>,
    //Receiver<Block>,
    Receiver<(Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>)>,
    Sender<Header>,
    Receiver<PrepareInfo>,
    Receiver<Certificate>,
    Sender<Certificate>,
    Store,
) {
    let (tx_core, rx_core) = channel(1);
    let (tx_loopback, rx_loopback) = channel(1);
    let (tx_loopback_cert, rx_loopback_cert) = channel(1);
    let (tx_proposer, rx_proposer) = channel(1);
    let (tx_mempool, mut rx_mempool) = channel(1);
    let (tx_commit, rx_commit) = channel(2);
    let (tx_consensus, rx_consensus) = channel(1);
    let (tx_validation, mut rx_validation) = channel(10);
    let (tx_ticket, rx_ticket) = channel(1);
    let (tx_special, rx_special) = channel(1);

    let(tx_loopback_process_commit, rx_loopback_process_commit) = channel(1);
    let(tx_loopback_commit, rx_loopback_commit) = channel(1);
    let(tx_request_header_sync, _rx_request_header_sync) = channel(1);

    let signature_service = SignatureService::new(secret);
    let _ = fs::remove_dir_all(store_path);
    let store = Store::new(store_path).unwrap();
    let leader_elector = LeaderElector::new(committee.clone());
    let mempool_driver = MempoolDriver::new(committee.clone(), tx_mempool);
    let synchronizer = Synchronizer::new(
        name,
        committee.clone(),
        store.clone(),
        tx_loopback,
        tx_loopback_cert,

        tx_request_header_sync,
        tx_loopback_process_commit,
        tx_loopback_commit,

        /* sync_retry_delay */ 100_000,
    );

    tokio::spawn(async move {
        loop {
            rx_mempool.recv().await;
        }
    });

    Core::spawn(
        name,
        committee,
        signature_service,
        store.clone(),
        leader_elector,
        mempool_driver,
        synchronizer,
        /* timeout_delay */ 100,
        /* rx_message */ rx_core,
        rx_consensus,
        rx_loopback,
        rx_loopback_cert,
        tx_proposer,
        tx_commit,
        tx_validation,
        tx_ticket,
        rx_special,
        rx_loopback_process_commit,
        rx_loopback_commit,
    );

    (tx_core, rx_validation, tx_special, rx_ticket, rx_commit, tx_consensus, store)
}

fn leader_keys(slot: Slot, view: View) -> (PublicKey, SecretKey) {
    let leader_elector = LeaderElector::new(committee());
    let leader = leader_elector.get_leader(slot, view);
    keys()
        .into_iter()
        .find(|(public_key, _)| *public_key == leader)
        .unwrap()
}

#[tokio::test]
#[serial]
async fn process_special_header() {
    let committee = committee_with_base_port(16_000);

    // Make a block and the vote we expect to receive.
    //let block = chain(vec![leader_keys(1)]).pop().unwrap();
    let (public_key, secret_key) = keys().pop().unwrap();
    let (public_key_1, secret_key_1) = leader_keys(1, 1);
    ////println!("Leader keys {}, {}, {}", leader_keys(2).0, leader_keys(3).0, leader_keys(4).0);
    //let vote = Vote::new_from_key(block.digest(), block.view, public_key, &secret_key);

    let ticket: Ticket = Ticket { header: Some(Header::genesis(&committee)), tc: None, slot: 0, proposals: BTreeMap::new() };
    let prepare_info: PrepareInfo = PrepareInfo { consensus_info: InstanceInfo { slot: 1, view: 1 }, ticket, proposals: HashMap::new() };
    let info_list = vec![prepare_info.clone()];

    let header = Header {author: public_key_1, height: 2, payload: BTreeMap::new(), parent_cert: Certificate::genesis_cert(&committee),
                         id: Digest::default(), signature: Signature::default(), prepare_info_list: info_list,
                         special_parent: None};
    let id = header.digest();
    let signature = Signature::new(&id, &secret_key_1);

    let head = Header {id, signature, ..header};
        //(public_key, 1, BTreeMap::new(), BTreeSet::new(), sig_service, true, 1, 1, None, 1).await;
    let validate: (Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>) = (head.clone(), vec![(prepare_info, true)], Vec::new());
    //let expected = bincode::serialize(&validate).unwrap();

    // Run a core instance.
    let store_path = ".db_test_handle_proposal";
    let (_, mut rx_validation, tx_special, _, _, _, _) =
        core(public_key, secret_key, committee.clone(), store_path);

    // Send a header to the core.
    //let message = ConsensusMessage::Propose(block.clone());
    tx_special.send(head).await.unwrap();

    let received = rx_validation.recv().await.unwrap();

    //assert!(received.0.eq(&validate.0));
    assert_eq!(received.1.get(0).unwrap().0.consensus_info, validate.1.get(0).unwrap().0.consensus_info);
    assert_eq!(received.1.get(0).unwrap().1, validate.1.get(0).unwrap().1);
    assert_eq!(received.2.is_empty(), validate.2.is_empty());
    //assert_eq!(received.3, validate.3);

    // Ensure the next leaders gets the vote.

    //let (next_leader, _) = leader_keys(1);
    //let address = committee.address(&next_leader).unwrap();
    //let handle = listener(address, Some(Bytes::from(expected)));
    //assert!(handle.await.is_ok());
}

#[tokio::test]
async fn process_special_cert() {
    let committee = committee_with_base_port(16_000);

    let (public_key, secret_key) = keys().pop().unwrap();
    let (public_key_1, secret_key_1) = leader_keys(1, 1);
    ////println!("Leader keys {}, {}, {}", leader_keys(2).0, leader_keys(3).0, leader_keys(4).0);
    //let vote = Vote::new_from_key(block.digest(), block.view, public_key, &secret_key);

    let ticket: Ticket = Ticket { header: Some(Header::genesis(&committee)), tc: None, slot: 0, proposals: BTreeMap::new() };
    let prepare_info: PrepareInfo = PrepareInfo { consensus_info: InstanceInfo { slot: 1, view: 1 }, ticket, proposals: HashMap::new() };
    let info_list = vec![prepare_info.clone()];

    let header = Header {author: public_key_1, height: 2, payload: BTreeMap::new(), parent_cert: Certificate::genesis_cert(&committee),
                         id: Digest::default(), signature: Signature::default(), prepare_info_list: info_list,
                         special_parent: None};
    let id = header.digest();
    let signature = Signature::new(&id, &secret_key_1);

    let head = Header {id, signature, ..header};
    let confirm_info: ConfirmInfo = ConfirmInfo { consensus_info: InstanceInfo { slot: 1, view: 1 }, cert_type: CertType::Prepare };

    let votes: Vec<_> = keys()
        .iter()
        .take(3)
        .map(|(public_key, secret_key)| {
            Vote::new_from_key(head.clone(), vec![(prepare_info.clone(), true)], Vec::new(), *public_key, &secret_key)
        })
        .map(|v| (v.author, v.signature))
        .collect();

    let certificate: Certificate = Certificate { author: head.origin(), header_digest: head.id.clone(), height: head.height(), valid_prepare_info: vec![(prepare_info, true)], valid_confirm_info: Vec::new(), votes, confirm_info_list: vec![confirm_info.clone()] };

    let header1 = Header {author: public_key_1, height: 3, payload: BTreeMap::new(), parent_cert: certificate,
                         id: Digest::default(), signature: Signature::default(), prepare_info_list: Vec::new(),
                         special_parent: None};
    let id1 = header1.digest();
    let signature1 = Signature::new(&id1, &secret_key_1);

    let head1 = Header {id: id1, signature: signature1, ..header1};

        //(public_key, 1, BTreeMap::new(), BTreeSet::new(), sig_service, true, 1, 1, None, 1).await;
    let validate: (Header, Vec<(PrepareInfo, bool)>, Vec<(ConfirmInfo, bool)>) = (head1.clone(), Vec::new(), vec![(confirm_info, true)]);
    //let expected = bincode::serialize(&validate).unwrap();

    // Run a core instance.
    let store_path = ".db_test_handle_proposal";
    let (_, mut rx_validation, tx_special, _, _, _, _) =
        core(public_key, secret_key, committee.clone(), store_path);

    // Send a header to the core.
    //let message = ConsensusMessage::Propose(block.clone());
    tx_special.send(head1).await.unwrap();

    let received = rx_validation.recv().await.unwrap();

    //assert!(received.0.eq(&validate.0));
    assert_eq!(received.2.get(0).unwrap().0.consensus_info, validate.2.get(0).unwrap().0.consensus_info);
    assert_eq!(received.2.get(0).unwrap().1, validate.2.get(0).unwrap().1);
    assert_eq!(received.1.is_empty(), validate.1.is_empty());
}

/*#[tokio::test]
async fn handle_accept_votes() {
    let committee = committee_with_base_port(16_000);

     // Make a header, cert and the accept_vote we expect to receive.
     let (public_key, secret_key) = keys().pop().unwrap();
     let (public_key_1, secret_key_1) = leader_keys(1);
     
     let ticket = Ticket::genesis(&committee);
 
     let header = Header {author: public_key_1, round: 2, payload: BTreeMap::new(), parents: BTreeSet::new(),
                          id: Digest::default(), signature: Signature::default(), is_special: true, view: 1,
                          prev_view_round: 0, special_parent: None, special_parent_round: 0, consensus_parent: Some(ticket.digest()), ticket: Some(ticket)};
     let id = header.digest();
     let signature = Signature::new(&id, &secret_key_1);
     let head = Header {id: id.clone(), signature, ..header};

     // Run a core instance.
     let store_path = ".db_test_handle_proposal";
     let (tx_core, _rx_commit, mut rx_validation, _, _, mut rx_commit, tx_consensus, mut store) =
         core(public_key, secret_key, committee.clone(), store_path);

     //Write header to store so it can be committed.
     let bytes = bincode::serialize(&head).expect("Failed to serialize header");
     store.write(head.id.to_vec(), bytes).await;

    //Create a set of votes to send to core
    let votes: Vec<_> = keys()
        .iter()
        .take(3)
        .map(|(public_key, secret_key)| {
            AcceptVote::new_from_key(id.clone(), 1, 1, *public_key, &secret_key)
        })
        .collect();

    //Create the QC we are expected to receive
    let high_qc = QC {
        hash: id,
        view: 1,
        view_round: 1,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: PublicKey::default(),
    };
    let expected = bincode::serialize(&ConsensusMessage::QC(high_qc)).unwrap();
    
    //create listeners to wait for QC
    let handles: Vec<_> = committee
    .broadcast_addresses(&public_key)
    .into_iter()
    .map(|(_, address)| listener(address, Some(Bytes::from(expected.clone()))))
    .collect();

    //Send votes to core
    for vote in votes.clone() {
        let message = ConsensusMessage::AcceptVote(vote);
        tx_core.send(message).await.unwrap();
    }
    //Receive Commit
    let b = rx_commit.recv().await.unwrap();

    //Ensure all listeners receive the QC
    assert!(try_join_all(handles).await.is_ok());
}


#[tokio::test]
#[serial]
async fn commit_header() {
    // Get enough distinct leaders to form a quorum.
    let leaders = vec![leader_keys(1), leader_keys(2), leader_keys(3)];
    //let chain = chain(leaders);

    // Run a core instance.
    let store_path = ".db_test_commit_block";
    //let mut store = Store::new(store_path).unwrap();

    let (public_key, secret_key) = keys().pop().unwrap();
    let (pub_key_1, priv_key_1) = leader_keys(1);

    let (tx_core, _, mut rx_validation, tx_special, mut rx_ticket, mut rx_commit, tx_consensus, mut store) =
        core(public_key, secret_key, committee(), store_path);

    let ticket = Ticket::genesis(&committee());

    let header = Header {author: pub_key_1, round: 1, payload: BTreeMap::new(), parents: BTreeSet::new(),
                         id: Digest::default(), signature: Signature::default(), is_special: true, view: 1,
                         prev_view_round: 0, special_parent: None, special_parent_round: 0, consensus_parent: Some(ticket.digest()), ticket: Some(ticket)};
    let id = header.digest();
    let signature = Signature::new(&id, &priv_key_1);

    let head = Header {id: id.clone(), signature, ..header};

    //Write header to store so it can be committed.
    let bytes = bincode::serialize(&head).expect("Failed to serialize header");
    store.write(head.id.to_vec(), bytes).await;

    //process_special_header //TODO: CAN REMOVE
    // tx_special.send(head.clone()).await.unwrap();
    // rx_validation.recv().await.unwrap();

    let votes: Vec<_> = keys()
        .iter()
        .take(3)
        .map(|(public_key, secret_key)| {
            AcceptVote::new_from_key(id.clone(), 1, 1, *public_key, &secret_key)
        })
        .collect();
    let high_qc = QC {
        hash: id,
        view: 1,
        view_round: 1,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: PublicKey::default()
    };

    let committed = Certificate { header: head, special_valids: Vec::new(), votes: high_qc.votes.clone() };

    //process special_cert  //TODO: CAN REMOVE
    //tx_consensus.send(committed.clone()).await.unwrap();

    //handle_qc
    let message = ConsensusMessage::QC(high_qc.clone());
    tx_core.send(message).await.unwrap();

    //TODO: Need to somehow add it to store. ==> Require Primary module for that. ==> Try to create a manual hack...


    //rx_validation.recv().await.unwrap();

    // Send a the blocks to the core.
    //let committed = chain[0].clone();
    //let committed = Certificate { header: head, special_valids: Vec::new(), votes: high_qc.votes.clone() };
    //for block in chain {
//        let message = ConsensusMessage::Propose(block);
  //      tx_core.send(message).await.unwrap();
    //}


    let b = rx_commit.recv().await.unwrap();
    assert_eq!(b, committed);
    // Ensure the core commits the head.
    /*match rx_commit.recv().await {
        Some(b) => assert_eq!(b, committed),
        _ => assert!(false),
    }*/
}

//NOTE: This code also tests the synchronizer code
#[tokio::test]
#[serial]
async fn commit_header_chain() {  
    // Get enough distinct leaders to form a quorum.
    let leaders = vec![leader_keys(1), leader_keys(2), leader_keys(3)];
    //let chain = chain(leaders);

    // Run a core instance.
    let store_path = ".db_test_commit_block";
    //let mut store = Store::new(store_path).unwrap();

    let (public_key, secret_key) = keys().pop().unwrap();
    let (pub_key_1, priv_key_1) = leader_keys(1);

    //Generating first header using genesis ticket
    let (tx_core, _, mut rx_validation, tx_special, mut rx_ticket, mut rx_commit, tx_consensus, mut store) =
        core(public_key, secret_key, committee(), store_path);

    let ticket = Ticket::genesis(&committee());

    let header = Header {author: pub_key_1, round: 1, payload: BTreeMap::new(), parents: BTreeSet::new(),
                         id: Digest::default(), signature: Signature::default(), is_special: true, view: 1,
                         prev_view_round: 0, special_parent: None, special_parent_round: 0, consensus_parent: Some(ticket.digest()), ticket: Some(ticket)};
    let id = header.digest();
    let signature = Signature::new(&id, &priv_key_1);
    let head = Header {id: id.clone(), signature, ..header};

    //Don't write first header to store yet.
    let head_1 = head.clone();

    //process_special_header //TODO: CAN REMOVE
    // tx_special.send(head.clone()).await.unwrap();
    // rx_validation.recv().await.unwrap();

    let votes: Vec<_> = keys()
        .iter()
        .take(3)
        .map(|(public_key, secret_key)| {
            AcceptVote::new_from_key(id.clone(), 1, 1, *public_key, &secret_key)
        })
        .collect();
    let high_qc = QC {
        hash: id,
        view: 1,
        view_round: 1,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: PublicKey::default(),
    };

    let committed_1 = Certificate { header: head.clone(), special_valids: Vec::new(), votes: high_qc.votes.clone() };
    // Note: don't need to process special cert first anymore
    //process special_cert  
    //tx_consensus.send(committed.clone()).await.unwrap();

    let ticket = Ticket::new(head.digest(), head.view.clone(), head.round.clone(), high_qc.clone(), None).await;

    //Generating second header, using previous QC as ticket.
    let (pub_key_1, priv_key_1) = leader_keys(1);
    let header = Header {author: pub_key_1, round: 2, payload: BTreeMap::new(), parents: BTreeSet::new(),
        id: Digest::default(), signature: Signature::default(), is_special: true, view: 2,
        prev_view_round: 1, special_parent: None, special_parent_round: 1, consensus_parent: Some(ticket.digest()), ticket: Some(ticket)};
    let id = header.digest();
    let signature = Signature::new(&id, &priv_key_1);
    let head = Header {id: id.clone(), signature, ..header};
    //Write header to store so it can be committed.
    let bytes = bincode::serialize(&head).expect("Failed to serialize header");
    store.write(head.id.to_vec(), bytes).await;

    //Generate QC
    let votes: Vec<_> = keys()
    .iter()
    .take(3)
    .map(|(public_key, secret_key)| {
        AcceptVote::new_from_key(id.clone(), 1, 1, *public_key, &secret_key)
    })
    .collect();
    let high_qc = QC {
        hash: id,
        view: 1,
        view_round: 1,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: PublicKey::default(),
    };
    let committed_2 = Certificate { header: head.clone(), special_valids: Vec::new(), votes: high_qc.votes.clone() };


    //Send QC for second header only. This should commit both headers.
    //handle_qc
    let message = ConsensusMessage::QC(high_qc.clone());
    tx_core.send(message).await.unwrap();

    //Write first header to store to wake up waiters --> this will allow the parent header to commit; which then wakes the waiter for the second header
    let bytes = bincode::serialize(&head_1).expect("Failed to serialize header");
    store.write(head_1.id.to_vec(), bytes).await;


    let b = rx_commit.recv().await.unwrap();
    assert_eq!(b, committed_1);
    let b = rx_commit.recv().await.unwrap();
    assert_eq!(b, committed_2);
   
}


#[tokio::test]
async fn generate_proposal() {
    // Get the keys of the leaders of this round and the next.
    let (leader, leader_key) = leader_keys(1);
    let (next_leader, next_leader_key) = leader_keys(2);

     // Run a core instance.
     let store_path = ".db_test_generate_proposal";
     let (tx_core, _, mut rx_validation, tx_special, mut rx_ticket, mut rx_commit, tx_consensus, mut store) =
         core(next_leader, next_leader_key, committee(), store_path);

    // Make a header, votes, and QC.
    let header = Header::new_from_key(leader, 1, 1, &leader_key, &committee()); //pubkey, view, round, privkey
    let id = header.digest();

   
    //Write header to store so it can be committed.
    let bytes = bincode::serialize(&header).expect("Failed to serialize header");
    store.write(header.id.to_vec(), bytes).await;

    let votes: Vec<_> = keys()
        .iter()
        .take(3)
        .map(|(public_key, secret_key)| {
            AcceptVote::new_from_key(id.clone(), header.view.clone(), header.round.clone(), *public_key, &secret_key)
        })
        .collect();
    let high_qc = QC {
        hash: id,
        view: header.view,
        view_round: header.round,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: PublicKey::default()
    };

    let message = ConsensusMessage::QC(high_qc.clone());
    tx_core.send(message).await.unwrap();

    // Ensure the core sends a new ticket.
    let ticket = rx_ticket.recv().await.unwrap();
    assert_eq!(ticket.round, 1);
    assert_eq!(ticket.view, 1);
    assert_eq!(ticket.qc, high_qc);
    assert!(ticket.tc.is_none());
}

#[tokio::test]
async fn commit_fast_qc() {
    // Get the keys of the leaders of this round and the next.
    let (leader, leader_key) = leader_keys(1);
    let (next_leader, next_leader_key) = leader_keys(2);

     // Run a core instance.
     let store_path = ".db_test_generate_proposal";
     let (tx_core, _, mut rx_validation, tx_special, mut rx_ticket, mut rx_commit, tx_consensus, mut store) =
         core(next_leader, next_leader_key, committee(), store_path);

    // Make a header, votes, and QC.
    let header = Header::new_from_key(leader, 1, 1, &leader_key, &committee()); //pubkey, view, round, privkey
    let id = header.digest();

   
    //Write header to store so it can be committed.
    let bytes = bincode::serialize(&header).expect("Failed to serialize header");
    store.write(header.id.to_vec(), bytes).await;

    let votes: Vec<Vote> = keys()
        .iter()
        .take(4)
        .map(|(public_key, secret_key)| {
            Vote::new_from_key(id.clone(), header.round.clone(), header.author.clone(), *public_key, &secret_key)
        })
        .collect();
    let fast_qc = QC {
        hash: id,
        view: header.view,
        view_round: header.round,
        votes: votes
            .iter()
            .cloned()
            .map(|x| (x.author, x.signature))
            .collect(),
        origin: header.author,
    };

    let message = ConsensusMessage::QC(fast_qc.clone());  //Fast qc
    tx_core.send(message).await.unwrap();

    // Ensure the core sends a new ticket.
    let ticket = rx_ticket.recv().await.unwrap();
    assert_eq!(ticket.round, 1);
    assert_eq!(ticket.view, 1);
    assert_eq!(ticket.qc, fast_qc);
    assert!(ticket.tc.is_none());
}

#[tokio::test]
async fn local_timeout_round() {
    let committee = committee_with_base_port(16_100);

    // Make the timeout vote we expect to send.
    let (public_key, secret_key) = leader_keys(3);
    let qc = QC::genesis(&committee);
    let timeout = Timeout::new_from_key(qc, 1, public_key, &secret_key);
    let expected = bincode::serialize(&ConsensusMessage::Timeout(timeout)).unwrap();

    // Run a core instance.
    let store_path = ".db_test_local_timeout_round";
    let (_tx_core, _rx_commit, _, _, _, _, _, _) =
        core(public_key, secret_key, committee.clone(), store_path);

    // Ensure the node broadcasts a timeout vote.
    let handles: Vec<_> = committee
        .broadcast_addresses(&public_key)
        .into_iter()
        .map(|(_, address)| listener(address, Some(Bytes::from(expected.clone()))))
        .collect();
    assert!(try_join_all(handles).await.is_ok());
}*/
