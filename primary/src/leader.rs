#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use crate::primary::{Slot, View};
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
        // TODO: Uncomment, this is strictly commented out for testing
        keys.sort();
        //let index = 2;
        let mut index = view + slot;
        // RR but skip partitioned node
        /*if index as usize % self.committee.size() == 0 {
            index += 1;
        }*/
        keys[index as usize % self.committee.size()]
        //keys[1]
    }
}
