use crate::consensus::{View, Slot};
use config::Committee;
use crypto::PublicKey;

//pub type LeaderElector = RRLeaderElector;
pub type LeaderElector = SemiParallelRRLeaderElector;


pub struct RRLeaderElector {
    committee: Committee,
}

impl RRLeaderElector {
    pub fn new(committee: Committee) -> Self {
        Self { committee }
    }

    pub fn get_leader(&self, view: View) -> PublicKey {
        let mut keys: Vec<_> = self.committee.authorities.keys().cloned().collect();
        keys.sort();
        keys[view as usize % self.committee.size()]
        //keys[0]
    }
}

pub struct SemiParallelRRLeaderElector {
    committee: Committee,
}

impl SemiParallelRRLeaderElector {
    pub fn new(committee: Committee) -> Self {
        Self { committee }
    }

    pub fn get_leader(&self, slot: Slot, view: View) -> PublicKey {
        let mut keys: Vec<_> = self.committee.authorities.keys().cloned().collect();
        keys.sort();
        let index = view + slot;
        keys[index as usize % self.committee.size()]
        //keys[0]
    }
}


