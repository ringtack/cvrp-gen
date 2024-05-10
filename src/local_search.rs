use std::time::Duration;

use rand::random;

use crate::Individual;

pub const EPSILON: f64 = 0.00001;
// Default limit on LS time.
pub const LS_LIMIT_SEC: u64 = 5;

pub const ACCEPT_TEMP: f64 = 0.1;
pub const MIN_ACCEPT_TEMP: f64 = 0.001;

/// Trait for a local search algorithm.
pub trait LocalSearch {
    /// Run local search.
    /// - `accept_temp`: temperature for SA to accept worse solution
    /// - `excess_penalty`: penalty for excess vehicle capacity
    /// - `time_limit`: time limit for search
    ///
    /// Returns the optimized individual found during local search.
    fn run(
        &mut self,
        time_limit: Duration,
        excess_penalty: f64,
        accept_temp: f64,
        min_accept_temp: f64,
    ) -> Individual;
}

/// Simulated annealing accept fn using temp.
pub fn sa_accept(accept_temp: f64) -> Box<dyn Fn(f64) -> bool> {
    Box::new(move |delta: f64| -> bool {
        // If delta negative, it lowers objective value, so accept
        if delta < -EPSILON {
            return true;
        } else {
            // Otherwise, accept with some probability accept_temp
            random::<f64>() < accept_temp
        }
    })
}

// /// Struct implementing local search algorithms: given an individual with
// some /// (potentially infeasible) solution, attempt to find better solutions.
// pub struct OLocalSearch {
//     /// Store best and current individual so far
//     best: Individual,
//     curr: Individual,
// }

// impl OLocalSearch {
//     /// Create a new local search instance from an individual.
//     pub fn new(ind: Individual) -> OLocalSearch {
//         OLocalSearch {
//             best: ind.clone(),
//             curr: ind,
//         }
//     }

//     /// Runs local search with the specified parameters, returning the best
//     /// solution found.
//     /// - `accept_temp`: temperature for SA to accept worse solution
//     /// - `time_limit`: time limit for search
//     pub fn run(
//         &mut self,
//         accept_temp: f64,
//         time_limit: Duration,
//         excess_penalty: f64,
//     ) -> Individual {
//         let mut hgs = HGSLS::new(self.curr.clone(), excess_penalty);

//         // TODO: explore actually using simulated annealing, rather than
// quitting on no         // improvement
//         let accept = sa_accept(accept_temp);

//         let start = Instant::now();
//         // Loop until time limit or no improvement
//         'ls: loop {
//             // If time limit exceeded, break
//             if start.elapsed() >= time_limit {
//                 log::debug!("Time limit exceeded");
//                 break 'ls;
//             }

//             // Performs a LS move to get a new solution
//             let res = hgs.ls_move(&accept);
//             // If succeeded, get the individual
//             if res {
//                 // let mut new_ind = hgs.get_individual();
//                 // // new_ind.objective();
//                 // // If new solution is better than current best, update
//                 // if new_ind.objective < self.best.objective {
//                 //     self.best = new_ind.clone();
//                 //     log::debug!(
//                 //         "Got better solution (old {}, new {})",
//                 //         self.best.objective,
//                 //         new_ind.objective
//                 //     );
//                 // }
//             } else {
//                 // If no improvement, break
//                 log::debug!("No improvement; stopping LS");
//                 self.best = hgs.get_individual();
//                 break 'ls;
//             }
//         }

//         // Return the best solution found
//         self.best.clone()
//     }
// }
