use super::*;
use crate::common::{committee_with_base_port, keys, listener, committee};
use crate::leader::LeaderElector;
use crate::consensus::Round;
use primary::messages::{Ticket, Header};
use crypto::{Signature, SecretKey, PublicKey};
use std::{collections::BTreeMap, collections::BTreeSet};
use crypto::Hash as _;
use std::fs;
use tokio::sync::mpsc::channel;

fn leader_keys(round: Round) -> (PublicKey, SecretKey) {
    let leader_elector = LeaderElector::new(committee());
    let leader = leader_elector.get_leader(1, round);
    keys()
        .into_iter()
        .find(|(public_key, _)| *public_key == leader)
        .unwrap()
}


//NOTE: THESE ARE DEPRECATED/NOT USED. We currently sync for headers through the DAG layer.

/*#[tokio::test]
async fn sync_reply() {
    let (tx_request, rx_request) = channel(1);
    let (tx_request_cert, rx_request_cert) = channel(1);

    let (requestor, _) = keys().pop().unwrap();
    let committee = committee_with_base_port(13_000);

    // Create a new test store.
    let path = ".db_test_sync_reply";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    let ticket = Ticket {hash: Header::genesis(&committee).id, qc: QC::genesis(&committee), tc: None, view: 0 , round: 0};

    let header = Header {author: leader_keys(1).0, round: 2, payload: BTreeMap::new(), parents: BTreeSet::new(),
                         id: Digest::default(), signature: Signature::default(), is_special: true, view: 1,
                         prev_view_round: 1, special_parent: None, special_parent_round: 0, consensus_parent: None, ticket: Some(ticket)};
    let id = header.digest();
    let signature = Signature::new(&id, &leader_keys(1).1);

    let head = Header {id: id.clone(), signature, ..header};


    // Add a batch to the store.
    let digest = head.digest();
    let serialized = bincode::serialize(&head).unwrap();
    store.write(digest.to_vec(), serialized.clone()).await;

    // Spawn an `Helper` instance.
    Helper::spawn(committee.clone(), store, rx_request, rx_request_cert);

    // Spawn a listener to receive the sync reply.
    let address = committee.address(&requestor).unwrap();
    let message = ConsensusMessage::Header(head);
    let expected = Bytes::from(bincode::serialize(&message).unwrap());
    let handle = listener(address, Some(expected));

    // Send a sync request.
    tx_request.send((digest, requestor)).await.unwrap();

    // Ensure the requestor received the batch (ie. it did not panic).
    assert!(handle.await.is_ok());
}*/
