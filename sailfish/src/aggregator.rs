use crate::consensus::View;
use primary::Certificate;
//use crate::error::{ConsensusError, ConsensusResult};
use primary::messages::{Timeout, TC};
use primary::{ensure, error::{ConsensusError, ConsensusResult}};
use config::{Committee, Stake};
use crypto::Hash as _;
use crypto::{Digest, PublicKey, Signature};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
#[path = "tests/aggregator_tests.rs"]
pub mod aggregator_tests;

pub struct Aggregator {
    committee: Committee,
    votes_aggregators: HashMap<View, HashMap<Digest, Box<QCMaker>>>,
    timeouts_aggregators: HashMap<View, Box<TCMaker>>,
}

impl Aggregator {
    pub fn new(committee: Committee) -> Self {
        Self {
            committee,
            votes_aggregators: HashMap::new(),
            timeouts_aggregators: HashMap::new(),
        }
    }

    /*pub fn add_vote(&mut self, vote: Vote) -> ConsensusResult<Option<QC>> {
        // TODO [issue #7]: A bad node may make us run out of memory by sending many votes
        // with different view numbers or different digests.

        // Add the new vote to our aggregator and see if we have a QC.
        self.votes_aggregators
            .entry(vote.view)
            .or_insert_with(HashMap::new)
            .entry(vote.digest())
            .or_insert_with(|| Box::new(QCMaker::new()))
            .append(vote, &self.committee)
    }*/

    /*pub fn add_timeout(&mut self, timeout: Timeout) -> ConsensusResult<Option<TC>> {
        // TODO: A bad node may make us run out of memory by sending many timeouts
        // with different view numbers.

        // Add the new timeout to our aggregator and see if we have a TC.
        self.timeouts_aggregators
            .entry(timeout.view)
            .or_insert_with(|| Box::new(TCMaker::new()))
            .append(timeout, &self.committee)
    }*/

    pub fn cleanup(&mut self, view: &View) {
        self.votes_aggregators.retain(|k, _| k >= view);
        self.timeouts_aggregators.retain(|k, _| k >= view);
    }
}

struct QCMaker {
    weight: Stake,
    votes: Vec<(PublicKey, Signature)>,
    used: HashSet<PublicKey>,
}

impl QCMaker {
    pub fn new() -> Self {
        Self {
            weight: 0,
            votes: Vec::new(),
            used: HashSet::new(),
        }
    }

    /* 
    /// Try to append a signature to a (partial) quorum.
    pub fn append(&mut self, vote: Vote, committee: &Committee) -> ConsensusResult<Option<QC>> {
        let author = vote.author;

        // Ensure it is the first time this authority votes.
        ensure!(
            self.used.insert(author),
            ConsensusError::AuthorityReuse(author)
        );

        self.votes.push((author, vote.signature.clone()));
        self.weight += committee.stake(&author);
        //println!("self weight is {}", self.votes.len());
        if self.weight >= committee.quorum_threshold() {
            self.weight = 0; // Ensures QC is only made once.
            return Ok(Some(QC {
                hash: vote.id.clone(),
                view: vote.view,
                view_round: vote.round, //Note: Currently aren't checking anywhere that the view_rounds of the Votes match. However, they must be matching transitively: Replicas only vote on certs, and a cert only is formed on matching view_round.
                votes: self.votes.clone(),
            }));
        }
        Ok(None)
    }
    */
}

struct TCMaker { //TimeoutCert for View Changes,
    weight: Stake,
 
    // In first iteration, Aggregator can simply be "dumb" and collect all timeouts. ==> In second iteration, can make it smarter if want to broadcast less data.
    votes: Vec<Timeout>, //In first iteration, just forward the timeout quorum. //TODO: In second iteration, send only quorum of signatures of hashes + 1 code proposal
    // votes: Vec<(PublicKey, Signature, View)>, // View == view of the highestQC a replica has seen.
    used: HashSet<PublicKey>,
}

impl TCMaker {
    pub fn new() -> Self {
        Self {
            weight: 0,
            votes: Vec::new(),
            used: HashSet::new(),
        }
    }

    /// Try to append a signature to a (partial) quorum.
    pub fn append(
        &mut self,
        timeout: Timeout,
        committee: &Committee,
    ) -> ConsensusResult<Option<TC>> {
        let author = timeout.author;

        // Ensure it is the first time this authority votes.
        ensure!(
            self.used.insert(author),
            ConsensusError::AuthorityReuse(author)
        );

        // Add the timeout to the accumulator.
        /*let view = timeout.view;
        self.votes
            .push(timeout);
        self.weight += committee.stake(&author);
        if self.weight >= committee.quorum_threshold() {
            self.weight = 0; // Ensures TC is only created once.
            let tc = TC::new(committee, view, self.votes.clone());
            return Ok(Some(tc));
        }*/
        Ok(None)
    }
}


// struct TCMaker { //TimeoutCert for View Changes,
//     weight: Stake,
//     votes: Vec<(PublicKey, Signature, View)>, // View == view of the highestQC a replica has seen.
//     used: HashSet<PublicKey>,
// }

// impl TCMaker {
//     pub fn new() -> Self {
//         Self {
//             weight: 0,
//             votes: Vec::new(),
//             used: HashSet::new(),
//         }
//     }

//     /// Try to append a signature to a (partial) quorum.
//     pub fn append(
//         &mut self,
//         timeout: Timeout,
//         committee: &Committee,
//     ) -> ConsensusResult<Option<TC>> {
//         let author = timeout.author;

//         // Ensure it is the first time this authority votes.
//         ensure!(
//             self.used.insert(author),
//             ConsensusError::AuthorityReuse(author)
//         );

//         // Add the timeout to the accumulator.
//         self.votes
//             .push((author, timeout.signature, timeout.high_qc.view));
//         self.weight += committee.stake(&author);
//         if self.weight >= committee.quorum_threshold() {
//             self.weight = 0; // Ensures TC is only created once.
//             return Ok(Some(TC {
//                 hash: Digest::default(), //TODO: FIXME: Replace this with our new TC rule: I.e. whatever logic picks a header to propose
//                 view: timeout.view,
//                 view_round: 0, //TODO: FIXME: Replace this with new TC rule (round of the header we propose)
//                 votes: self.votes.clone(),
//             }));
//         }
//         Ok(None)
//     }
// }
