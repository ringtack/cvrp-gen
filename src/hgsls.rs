use std::time::{Duration, Instant};

use rand::{seq::SliceRandom, thread_rng};

use crate::{
    local_search::{sa_accept, LocalSearch, EPSILON},
    polar_sector::PolarSector,
    vrp_instance::Customer,
    Individual,
};

/// Type alias for an acceptance operator function: takes in the delta in
/// objective, and returns whether to accept it.
pub type AcceptOperator = Box<dyn Fn(f64) -> bool>;

/// A local search operator selector following the scheme in the HGS-CVRP paper:
/// - Shuffle order in which customers, then routes, are processed
/// - For each customer, shuffle its n nearest neighbors
/// - For each customer i:
///   - for each NN j: if one of the following moves yields an objective
///     increase, apply it:
///     - Relocate i to after j
///     - Relocate (i, i_next) to after j
///     - Relocate (i, i_next) to after j as (i_next, i)
///     - Swap i with j
///     - Swap (i, i_next) with j
///     - Swap (i, i_next) with (j, j_next)
///     - 2-opt (if r_i == r_j): swap (i, i_next) and (j, j_next) with (i,
///       j_next) and (j, i_next)
///     - 2-opt* (if r_i != r_j): swap (i, i_next) and (j, j_next) with (i, j)
///       and (i_next, j_next)
///     - 2-opt* (if r_i != r_j): swap (i, i_next) and (j, j_next) with (i,
///       j_next) and (j, i_next)
pub struct HGSLS {
    /// HGS LS internal structures
    ///
    /// Order in which to process customers
    cust_order: Vec<usize>,
    /// Shuffled nearest neighbors for each customer
    nn_order: Vec<Vec<Customer>>,
    /// Order in which to process vehicle routes
    route_order: Vec<usize>,

    /// Customers i and j on which operators will act
    cust_i: usize,
    cust_j: usize,

    /// Individual being optimized
    ind: Individual,
    /// Capacity penalty
    excess_penalty: f64,
    // /// Pred/succs for each customer
    // pred: Vec<usize>,
    // succ: Vec<usize>,
    // /// Routes for each customer
    // cust_routes: Vec<usize>,
    // /// Vehicle loads
    // loads: Vec<f64>,
}

impl HGSLS {
    /// Initializes internal HGSLS structures for a given individual.
    pub fn new(ind: Individual) -> HGSLS {
        // Initialize HGSLS values from individual
        let mut hgsls = HGSLS {
            cust_order: ind.total_route.clone(),
            nn_order: ind.vrp.customer_nns.clone(),
            route_order: (0..ind.vrp.n_vehicles).collect(),
            cust_i: 0,
            cust_j: 0,
            ind,
            excess_penalty: 0.,
        };
        // Initialize (i.e. shuffle) internal structures
        hgsls.reset();
        hgsls
    }

    /// (Re-)initializes the HGSLS internal structures. Assumes each one is
    /// already filled; thus, in the constructor, make sure to set values
    /// before calling this method.
    fn reset(&mut self) {
        self.cust_i = 0;
        self.cust_j = 0;
        // Shuffle customer, route, and NN order
        self.cust_order.shuffle(&mut thread_rng());
        self.route_order.shuffle(&mut thread_rng());
        for nns in self.nn_order.iter_mut() {
            nns.shuffle(&mut thread_rng());
        }

        log::trace!("Customer order: {:?}", self.cust_order);
    }

    /// If accepted, relocate i to after j
    fn relocate_i_j(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying relocate of {} to after {}", i, j);

        // If j is depot or i == succ_j, skip
        if j == 0 || i == self.ind.succ[j] {
            log::trace!("Skipping relocate; {} is depot or {} == succ j", j, i);
            return false;
        }

        // Get pred/succ for i and j
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let succ_j = self.ind.succ[j];

        log::trace!("pred_i: {}, succ_i: {}, succ_j: {}", pred_i, succ_i, succ_j);

        // Check distance delta for relocation:
        // - for i's route: d(pred_i, succ_i) - d(pred_i, i) - d(i, succ_i)
        // - for j's route: d(j, i) + d(i, succ_j) - d(j, succ_j)
        let dist_mtx = &self.ind.vrp.dist_mtx;
        let mut delta_i = dist_mtx[pred_i][succ_i] - dist_mtx[pred_i][i] - dist_mtx[i][succ_i];
        let mut delta_j = dist_mtx[j][i] + dist_mtx[i][succ_j] - dist_mtx[j][succ_j];

        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            log::trace!("inter route swap");
            let load_i = self.ind.vrp.customers[i].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i += self.load_penalty(self.ind.loads[ri] - load_i) - penalty_i;
            delta_j += self.load_penalty(self.ind.loads[rj] + load_i) - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!(
                "Relocating {} after {} (d_i: {}, d_j: {})",
                i,
                j,
                // net_change,
                delta_i,
                delta_j
            );

            // ugh my indexing here is ugly but oh well
            self.insert_after(j, i);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, relocate (i, i_next) to after j
    fn relocate_iin_j(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying relocate ({}, {}) after {}", i, self.ind.succ[i], j);

        log::trace!("i route: {:?}", self.ind.routes[self.ind.cust_routes[i]]);
        log::trace!(
            "i pred, succ, succsucc: {}, {}, {}",
            self.ind.pred[i],
            self.ind.succ[i],
            self.ind.succ[self.ind.succ[i]]
        );
        // Get pred/succ for i, i_next, and j
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let succ_succ_i = self.ind.succ[succ_i];
        let succ_j = self.ind.succ[j];

        // If i == succ_j, or succ_i == j, or succ_i is depot, skip
        if succ_j == i || succ_i == j || succ_i == 0 || j == 0 {
            log::trace!(
                "Skipping relocate; {} and {} adjacent or next/j is depot",
                i,
                j
            );
            return false;
        }

        // Check distance delta for relocation:
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta_i = dm[pred_i][succ_succ_i] - dm[pred_i][i] - dm[succ_i][succ_succ_i];
        let mut delta_j = dm[j][i] + dm[succ_i][succ_j] - dm[j][succ_j];
        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            let load_i = self.ind.vrp.customers[i].demand;
            let load_i_next = self.ind.vrp.customers[succ_i].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i += self.load_penalty(self.ind.loads[ri] - load_i - load_i_next) - penalty_i;
            delta_j += self.load_penalty(self.ind.loads[rj] + load_i + load_i_next) - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!(
                "Relocating ({}, {}) after {} (delta {})",
                i,
                self.ind.succ[i],
                j,
                net_change
            );
            // ugh my indexing here is ugly but oh well
            self.insert_after(j, i);
            self.insert_after(i, succ_i);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, relocate (i, i_next) to after j as (i_next, i)
    fn relocate_iin_j_flip(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!(
            "Trying relocate ({}, {}) after {} as ({}, {})",
            i,
            self.ind.succ[i],
            j,
            self.ind.succ[i],
            i
        );

        // Get pred/succ for i, i_next, and j
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let succ_i_next = self.ind.succ[succ_i];
        let succ_j = self.ind.succ[j];

        // If i == succ_j, or succ_i == j, or succ_i is depot, skip
        if succ_j == i || succ_i == j || succ_i == 0 || j == 0 {
            log::trace!(
                "Skipping relocate; {} and {} adjacent or next/j is depot",
                i,
                j
            );
            return false;
        }

        // Check distance delta for relocation:
        // - for i's route: d(pred_i, succ_i) - d(pred_i, i) - d(i, succ_i)
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta_i =
            dm[pred_i][succ_i_next] - dm[pred_i][i] - dm[i][succ_i] - dm[succ_i][succ_i_next];
        let mut delta_j = dm[j][succ_i] + dm[succ_i][i] + dm[i][succ_j] - dm[j][succ_j];
        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            let load_i = self.ind.vrp.customers[i].demand;
            let load_i_next = self.ind.vrp.customers[succ_i].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i += self.load_penalty(self.ind.loads[ri] - load_i - load_i_next) - penalty_i;
            delta_j += self.load_penalty(self.ind.loads[rj] + load_i + load_i_next) - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!(
                "Relocating ({}, {}) after {} as ({}, {}) (delta {})",
                i,
                self.ind.succ[i],
                j,
                self.ind.succ[i],
                i,
                net_change
            );
            self.insert_after(j, succ_i);
            self.insert_after(succ_i, i);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, swap i with j
    fn swap_i_j(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying swap {} with {}", i, j);

        // Get pred/succ for i and j
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let pred_j = self.ind.pred[j];
        let succ_j = self.ind.succ[j];

        // If i == pred_j or i == succ_j, skip
        if i == pred_j || i == succ_j {
            log::trace!("Skipping swap; {} and {} adjacent", i, j);
            return false;
        }

        // Check distance delta for swap:
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta_i = dm[pred_i][j] + dm[j][succ_i] - dm[pred_i][i] - dm[i][succ_i];
        let mut delta_j = dm[pred_j][i] + dm[i][succ_j] - dm[pred_j][j] - dm[j][succ_j];

        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            let load_i = self.ind.vrp.customers[i].demand;
            let load_j = self.ind.vrp.customers[j].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i += self.load_penalty(self.ind.loads[ri] - load_i + load_j) - penalty_i;
            delta_j += self.load_penalty(self.ind.loads[rj] - load_j + load_i) - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!("Swapping {} and {} (delta {})", i, j, net_change);
            self.swap(i, j);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, swap (i, i_next) with j
    fn swap_iin_j(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying swap ({}, {}) with {}", i, self.ind.succ[i], j);

        // Get pred/succ for i, i_next, and j
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let succ_succ_i = self.ind.succ[succ_i];
        let pred_j = self.ind.pred[j];
        let succ_j = self.ind.succ[j];

        // If i == pred_j or i_next == pred_j or i == succ_j or succ_i == depot, skip
        if i == pred_j || succ_i == pred_j || i == succ_j || succ_i == 0 {
            log::trace!("Skipping swap; {} and {} adjacent or next is depot", i, j);
            return false;
        }

        // Compute distance delta for swap:
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta_i =
            dm[pred_i][j] + dm[j][succ_succ_i] - dm[pred_i][i] - dm[succ_i][succ_succ_i];
        let mut delta_j = dm[pred_j][i] + dm[succ_i][succ_j] - dm[pred_j][j] - dm[j][succ_j];

        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            let load_i = self.ind.vrp.customers[i].demand;
            let load_i_next = self.ind.vrp.customers[succ_i].demand;
            let load_j = self.ind.vrp.customers[j].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i +=
                self.load_penalty(self.ind.loads[ri] - load_i - load_i_next + load_j) - penalty_i;
            delta_j +=
                self.load_penalty(self.ind.loads[rj] - load_j + load_i + load_i_next) - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!(
                "Swapping ({}, {}) with {} (delta {})",
                i,
                self.ind.succ[i],
                j,
                net_change
            );
            self.swap(i, j);
            self.insert_after(i, succ_i);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, swap (i, i_next) with (j, j_next)
    fn swap_iin_jjn(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!(
            "Trying swap ({}, {}) with ({}, {})",
            i,
            self.ind.succ[i],
            j,
            self.ind.succ[j]
        );

        // Get pred/succ for i, i_next, j, and j_next
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let succ_succ_i = self.ind.succ[succ_i];
        let pred_j = self.ind.pred[j];
        let succ_j = self.ind.succ[j];
        let succ_succ_j = self.ind.succ[succ_j];

        // If succ_i or succ_j is depot, or succ_j == pred_i or i == succ_j or succ_i ==
        // j or j == succ_succ_i, skip
        if succ_i == 0
            || succ_j == 0
            || succ_j == pred_i
            || i == succ_j
            || succ_i == j
            || j == succ_succ_i
        {
            log::trace!("Skipping swap; {} and {} adjacent or next is depot", i, j);
            return false;
        }

        // Compute distance delta for swap:
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta_i =
            dm[pred_i][j] + dm[succ_j][succ_succ_i] - dm[pred_i][i] - dm[succ_i][succ_succ_i];
        let mut delta_j =
            dm[pred_j][i] + dm[succ_i][succ_succ_j] - dm[pred_j][j] - dm[succ_j][succ_succ_j];

        // If swapping across routes, record change in vehicle load
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            let load_i = self.ind.vrp.customers[i].demand;
            let load_i_next = self.ind.vrp.customers[succ_i].demand;
            let load_j = self.ind.vrp.customers[j].demand;
            let load_j_next = self.ind.vrp.customers[succ_j].demand;

            // See if load exceeds capacity in either swap
            let penalty_i = self.load_penalty(self.ind.loads[ri]);
            let penalty_j = self.load_penalty(self.ind.loads[rj]);
            delta_i += self
                .load_penalty(self.ind.loads[ri] - load_i - load_i_next + load_j + load_j_next)
                - penalty_i;
            delta_j += self
                .load_penalty(self.ind.loads[rj] - load_j - load_j_next + load_i + load_i_next)
                - penalty_j;
        }

        let net_change = delta_i + delta_j;
        if accept(net_change) {
            log::trace!(
                "Swapping ({}, {}) with ({}, {}) (delta {})",
                i,
                self.ind.succ[i],
                j,
                self.ind.succ[j],
                net_change
            );
            self.swap(i, j);
            self.swap(succ_i, succ_j);

            self.ind.objective += net_change;
            return true;
        }
        false
    }

    /// If accepted, apply 2-opt move
    fn two_opt(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying 2-opt move on {} and {}", i, j);

        // Only apply if on same route
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri != rj {
            log::trace!("Skipping 2-opt move; {} and {} on different routes", i, j);
            return false;
        }
        // If less than 3 customers, skip
        let route = &mut self.ind.routes[ri];
        if route.len() < 3 {
            log::trace!(
                "Skipping 2-opt move; route {} has less than 3 customers",
                ri
            );
            return false;
        }

        // Get succ for i and j
        let succ_i = self.ind.succ[i];
        let succ_j = self.ind.succ[j];

        // Compute distance delta for 2-opt swap:
        let dm = &self.ind.vrp.dist_mtx;
        let delta = dm[i][j] + dm[succ_i][succ_j] - dm[i][succ_i] - dm[j][succ_j];

        if accept(delta) {
            log::trace!("Applying 2-opt move on {} and {} (delta {})", i, j, delta);

            // Find indices of i and j (reverse =j)
            let idx_i = route.iter().position(|&x| x == i).unwrap_or_default();
            let idx_j = route.iter().position(|&x| x == j).unwrap_or_default();
            // Sort
            let (idx_i, idx_j) = if idx_i < idx_j {
                (idx_i, idx_j)
            } else {
                (idx_j, idx_i)
            };

            // NOTE: this doesn't work
            // Specially preds/succs of i, j, succ_i, and succ_j
            // self.ind.succ[i] = j;
            // self.ind.pred[j] = i;
            // self.ind.succ[succ_i] = succ_j;
            // self.ind.pred[succ_j] = succ_i;

            // Reverse the segment between i and =j
            route[(idx_i + 1)..=idx_j].reverse();

            // Re-assign preds/succs for every customer in route
            // TODO: optimize this such that we only need to update preds/succs for reversed
            // parts
            for k in 0..route.len() {
                let c = route[k];
                if k == 0 {
                    self.ind.pred[c] = 0;
                } else {
                    self.ind.pred[c] = route[k - 1];
                }
                if k == route.len() - 1 {
                    self.ind.succ[c] = 0;
                } else {
                    self.ind.succ[c] = route[k + 1];
                }
            }

            self.ind.objective += delta;
            return true;
        }
        false
    }

    /// If accepted, apply 2-opt* move (swap (i, i_next) and (j, j_next) with
    /// (i, j_next) and (j, i_next))
    fn two_opt_star(&mut self, accept: &AcceptOperator) -> bool {
        let i = self.cust_i;
        let j = self.cust_j;

        log::trace!("Trying 2-opt* move on {} and {}", i, j);

        // Get succ for i and j
        let succ_i = self.ind.succ[i];
        let succ_j = self.ind.succ[j];

        // Only apply if on different routes
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri == rj {
            log::trace!("Skipping 2-opt* move; {} and {} on same route", i, j);
            return false;
        }

        // Compute distance delta for 2-opt* swap
        let dm = &self.ind.vrp.dist_mtx;
        let mut delta = dm[i][succ_j] + dm[j][succ_i] - dm[i][succ_i] - dm[j][succ_j];

        let mut route_i = self.ind.routes[ri].clone();
        let mut route_j = self.ind.routes[rj].clone();
        let idx_i = route_i.iter().position(|&x| x == i).unwrap_or_default();
        let idx_j = route_j.iter().position(|&x| x == j).unwrap_or_default();

        // Compute change in load: all customers after i are moved to j's route, and
        // vice versa
        let mut moved_load_ri = 0.;
        for k in route_i[(idx_i + 1)..].iter() {
            moved_load_ri += self.ind.vrp.customers[*k].demand;
        }
        let mut moved_load_rj = 0.;
        for k in route_j[(idx_j + 1)..].iter() {
            moved_load_rj += self.ind.vrp.customers[*k].demand;
        }

        let penalty_i = self.load_penalty(self.ind.loads[ri]);
        let penalty_j = self.load_penalty(self.ind.loads[rj]);
        delta += self.load_penalty(self.ind.loads[ri] - moved_load_ri + moved_load_rj) - penalty_i;
        delta += self.load_penalty(self.ind.loads[rj] - moved_load_rj + moved_load_ri) - penalty_j;

        if accept(delta) {
            log::trace!("Applying 2-opt* move on {} and {} (delta {})", i, j, delta);

            // If accept, then move [succ_j:] to after i in route i, and [succ_i:] to after
            // j in route j

            // Drain idx_i+1.. from route i, and idx_j+1 from route j
            let after_i = route_i.drain((idx_i + 1)..).collect::<Vec<_>>();
            let after_j = route_j.drain((idx_j + 1)..).collect::<Vec<_>>();
            route_i.extend(after_j);
            route_j.extend(after_i);

            // Re-compute preds/succs/routes/loads for the each customer in the two routes
            self.ind.loads[ri] = 0.;
            for i in 0..route_i.len() {
                let c = route_i[i];
                if i == 0 {
                    self.ind.pred[c] = 0;
                } else {
                    self.ind.pred[c] = route_i[i - 1];
                }
                if i == route_i.len() - 1 {
                    self.ind.succ[c] = 0;
                } else {
                    self.ind.succ[c] = route_i[i + 1];
                }
                self.ind.cust_routes[c] = ri;
                self.ind.loads[ri] += self.ind.vrp.customers[c].demand;
            }
            self.ind.loads[rj] = 0.;
            for j in 0..route_j.len() {
                let c = route_j[j];
                if j == 0 {
                    self.ind.pred[c] = 0;
                } else {
                    self.ind.pred[c] = route_j[j - 1];
                }
                if j == route_j.len() - 1 {
                    self.ind.succ[c] = 0;
                } else {
                    self.ind.succ[c] = route_j[j + 1];
                }
                self.ind.cust_routes[c] = rj;
                self.ind.loads[rj] += self.ind.vrp.customers[c].demand;
            }

            log::trace!(
                "before swapping routes: {:?}, {:?}",
                self.ind.routes[ri],
                self.ind.routes[rj]
            );
            self.ind.routes[ri] = route_i;
            self.ind.routes[rj] = route_j;

            self.ind.objective += delta;

            log::trace!(
                "after swapping routes: {:?}, {:?}",
                self.ind.routes[ri],
                self.ind.routes[rj]
            );
            log::trace!("preds: {:?}", self.ind.pred);
            log::trace!("succs: {:?}", self.ind.succ);
            return true;
        }

        false
    }

    /// If accepted, apply 2-opt* move with flipped order
    // fn two_opt_star_flip(&mut self, accept: &AcceptOperator) -> bool {
    //     // TODO: actually implement
    //     return false;
    // }

    // /// Attempt to apply swap* operator, swapping routes based on polar sectors.
    fn swap_star(&mut self, _accept: &AcceptOperator) -> bool {
        // TODO: actually implement
        return false;
    }

    /// Get the route id associated with the route index

    /// Computes the excess penalty for a load
    fn load_penalty(&self, load: f64) -> f64 {
        (load - self.ind.vrp.vehicle_cap as f64).max(0.) * self.excess_penalty
    }

    /// Inserts customer w/ id j after customer w/ id i
    fn insert_after(&mut self, i: usize, j: usize) {
        let succ_i = self.ind.succ[i];
        let pred_j = self.ind.pred[j];
        let succ_j = self.ind.succ[j];

        // Update pred/succ for i and j
        self.ind.succ[i] = j;
        self.ind.pred[succ_i] = j;
        self.ind.succ[j] = succ_i;
        self.ind.pred[j] = i;
        self.ind.succ[pred_j] = succ_j;
        self.ind.pred[succ_j] = pred_j;

        // Remove j from its route
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        if ri == rj {
            let route = &mut self.ind.routes[rj];
            log::trace!("Moving {} to after {} in route {:?}", j, i, route);
            let idx = route.iter().position(|&x| x == j).unwrap();
            route.remove(idx);
            // Insert j after i in i's route
            let idx = route.iter().position(|&x| x == i).unwrap();
            route.insert(idx + 1, j);
        } else {
            // Update loads for both routes
            let load_j = self.ind.vrp.customers[j].demand;
            self.ind.loads[rj] -= load_j;
            self.ind.loads[ri] += load_j;
            // Change routes of j
            self.ind.cust_routes[j] = ri;

            // Remove j from rj and insert after i in ri
            let route_j = &mut self.ind.routes[rj];
            log::trace!("Removing {} from route {:?}", j, route_j);
            let idx = route_j.iter().position(|&x| x == j).unwrap();
            route_j.remove(idx);
            let route_i = &mut self.ind.routes[ri];
            log::trace!("Inserting {} after {} in route {:?}", j, i, route_i);
            let idx = route_i.iter().position(|&x| x == i).unwrap();
            route_i.insert(idx + 1, j);
        }

        log::trace!(
            "After insert, ri: {:?}, rj: {:?}",
            self.ind.routes[ri],
            self.ind.routes[rj]
        );

        log::trace!("{i} pred, succ: {}, {}", self.ind.pred[i], self.ind.succ[i]);
        log::trace!("{j} pred, succ: {}, {}", self.ind.pred[j], self.ind.succ[j]);
        log::trace!(
            "{succ_i} pred, succ: {}, {}",
            self.ind.pred[succ_i],
            self.ind.succ[succ_i]
        );
        log::trace!(
            "{pred_j} pred, succ: {}, {}",
            self.ind.pred[pred_j],
            self.ind.succ[pred_j]
        );
    }

    /// Swaps customer w/ id i with customer w/ id j
    fn swap(&mut self, i: usize, j: usize) {
        let pred_i = self.ind.pred[i];
        let succ_i = self.ind.succ[i];
        let pred_j = self.ind.pred[j];
        let succ_j = self.ind.succ[j];

        // Update pred/succ for i and j
        self.ind.succ[pred_i] = j;
        self.ind.pred[succ_i] = j;
        self.ind.succ[pred_j] = i;
        self.ind.pred[succ_j] = i;
        self.ind.pred[i] = pred_j;
        self.ind.succ[i] = succ_j;
        self.ind.pred[j] = pred_i;
        self.ind.succ[j] = succ_i;

        // Update routes for i and j
        let ri = self.ind.cust_routes[i];
        let rj = self.ind.cust_routes[j];
        // If same route, swap in place
        if ri == rj {
            let route = &mut self.ind.routes[ri];
            let idx_i = route.iter().position(|&x| x == i).unwrap();
            let idx_j = route.iter().position(|&x| x == j).unwrap();
            route.swap(idx_i, idx_j);
        } else {
            // Update loads for both routes
            let load_i = self.ind.vrp.customers[i].demand;
            let load_j = self.ind.vrp.customers[j].demand;
            self.ind.loads[ri] += load_j - load_i;
            self.ind.loads[rj] += load_i - load_j;
            // Change routes of i and j
            self.ind.cust_routes[i] = rj;
            self.ind.cust_routes[j] = ri;

            // Replace i from ri w/ j
            let route_i = &mut self.ind.routes[ri];
            let idx_i = route_i.iter().position(|&x| x == i).unwrap();
            route_i[idx_i] = j;
            // Replace j from rj w/ i
            let route_j = &mut self.ind.routes[rj];
            let idx_j = route_j.iter().position(|&x| x == j).unwrap();
            route_j[idx_j] = i;
        }
    }
}

impl LocalSearch for HGSLS {
    fn run(&mut self, time_limit: Duration, excess_penalty: f64, accept_temp: f64) -> Individual {
        // Actually improved vs total calls
        // let mut stats = vec![(0, 0); 8];

        // Set excess penalty (used by HGSLS for delta computation, used by Individual
        // for objective computation)
        self.excess_penalty = excess_penalty;
        self.ind.set_excess_penalty(excess_penalty);

        // Get accept operator
        let accept = sa_accept(accept_temp);

        let mut improved = true;
        let start = Instant::now();
        // Loop until no improvement
        while improved {
            improved = false;
            // Re-initialize metadata
            log::trace!("Resetting HGSLS w/ new random order");
            self.reset();

            // If surpassed time limit, break
            if start.elapsed() >= time_limit {
                log::debug!("reached time limit; breaking out of LS");
                break;
            }

            let mut ci = 0;
            let mut cj = 0;
            let mut ri = 0;
            let mut rj = 0;

            // For each customer, try all NNs
            while ci < self.cust_order.len() {
                let cust_i = self.cust_order[ci];
                // If still have NN to do, process by sequentially trying any of the operators
                while cj < self.nn_order[cust_i].len() {
                    let cust_j = self.nn_order[cust_i][cj].id;
                    log::trace!(
                        "Processing customer num {}'s NN num {} (ci: {}, cj: {})",
                        cust_i,
                        cust_j,
                        ci,
                        cj
                    );
                    // let obj = self.ind.objective();
                    // let mut pre = self.ind.clone();

                    self.cust_i = cust_i;
                    self.cust_j = cust_j;
                    if self.relocate_i_j(&accept) {
                        improved = true;
                        // if self.ind.objective() <= obj + EPSILON {
                        //     stats[0].0 += 1;
                        // }
                        // stats[0].1 += 1;
                    } else if self.relocate_iin_j(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[1].0 += 1;
                        // }
                        // stats[1].1 += 1;
                    } else if self.relocate_iin_j_flip(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[2].0 += 1;
                        // }
                        // stats[2].1 += 1;
                    } else if self.swap_i_j(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[3].0 += 1;
                        // }
                        // stats[3].1 += 1;
                    } else if self.swap_iin_j(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[4].0 += 1;
                        // }
                        // stats[4].1 += 1;
                    } else if self.swap_iin_jjn(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[5].0 += 1;
                        // }
                        // stats[5].1 += 1;
                    } else if self.two_opt(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[6].0 += 1;
                        // }
                        // stats[6].1 += 1;
                    } else if self.two_opt_star(&accept) {
                        improved = true;
                        // if self.ind.objective() < obj {
                        //     stats[7].0 += 1;
                        // }
                        // stats[7].1 += 1;
                    } else if self.ind.pred[cust_j] == 0 {
                        // If cj's predecessor is depot, try moves that insert
                        // ci right after depot
                        // log::trace!("cust {}'s pred is depot", cust_j);
                        // self.cust_j = 0;
                        // if self.relocate_i_j(&accept) {
                        //     improved = true;
                        // } else if self.relocate_iin_j(&accept) {
                        //     improved = true;
                        // } else if self.relocate_iin_j_flip(&accept) {
                        //     improved = true;
                        // } else if self.two_opt_star(&accept) {
                        //     improved = true;
                        // }
                    }

                    // Regardless of success/failure, increment cj to next closest NN
                    cj += 1;
                }
                // Go to next customer and reset cj
                ci += 1;
                cj = 0;
            }

            // For each route pair, try swaps
            while ri < self.route_order.len() {
                // If have another route to do, process with swap*
                while rj < self.route_order.len() {
                    // If ri == rj, continue
                    if true || ri == rj {
                        rj += 1;
                        continue;
                    }

                    // If either route is empty, skip
                    let route_i = self.route_order[ri];
                    let route_j = self.route_order[rj];
                    if self.ind.routes[route_i].len() == 0 || self.ind.routes[route_j].len() == 0 {
                        rj += 1;
                        continue;
                    }

                    log::trace!(
                        "Processing swap* for route nums {} and {}",
                        route_i,
                        route_j,
                    );
                    log::trace!(
                        "Route {} customers: {:?}",
                        route_i,
                        self.ind.routes[route_i],
                    );
                    log::trace!(
                        "Route {} customers: {:?}",
                        route_j,
                        self.ind.routes[route_j],
                    );

                    // Check if polar sectors overlap
                    let polars_i = self.ind.routes[route_i]
                        .iter()
                        .map(|&c| self.ind.vrp.customers[c].polar_angle)
                        .collect::<Vec<_>>();
                    let polars_j = self.ind.routes[route_j]
                        .iter()
                        .map(|&c| self.ind.vrp.customers[c].polar_angle)
                        .collect::<Vec<_>>();
                    let psi = PolarSector::from_points(&polars_i);
                    let psj = PolarSector::from_points(&polars_j);
                    if psi.overlaps(&psj) {
                        if self.swap_star(&accept) {
                            improved = true;
                            rj += 1;
                        }
                    }
                    rj += 1;
                }
                // Otherwise, go to next route and reset rj
                ri += 1;
                rj = 0;
            }
        }

        // log::info!("HGSLS stats: {:?}", stats);
        // Re-initialize metadata, then return
        self.ind.initialize_metadata();
        self.ind.clone()
    }
}
