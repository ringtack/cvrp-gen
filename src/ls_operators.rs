use std::net;
use std::sync::Arc;

use rand::thread_rng;
use rand::seq::SliceRandom;

use crate::polar_sector::PolarSector;
use crate::vrp_instance::{Customer, VRPInstance};
use crate::Individual;

/// Type alias for an acceptance operator function: takes in the delta in objective, and returns
/// whether to accept it.
pub type AcceptOperator = Box<dyn Fn(f64) -> bool>;

/// Trait for a meta-heuristic operator application scheme.
pub trait MHOperator {
  /// Potentially applies an operator to an individual, returning whether the operator was applied.
  fn ls_move(&mut self, accept: &AcceptOperator) -> bool;

  /// Gets the individual from the operator.
  fn get_individual(&self) -> Individual;
}

/// A local search operator selector following the scheme in the HGS-CVRP paper:
/// - Shuffle order in which customers, then routes, are processed
/// - For each customer, shuffle its n nearest neighbors
/// - For each customer i:
///   - for each NN j: if one of the following moves yields an objective increase, apply it:
///     - Relocate i to after j
///     - Relocate (i, i_next) to after j
///     - Relocate (i, i_next) to after j as (i_next, i)
///     - Swap i with j
///     - Swap (i, i_next) with j
///     - Swap (i, i_next) with (j, j_next)
///     - 2-opt (if r_i == r_j): swap (i, i_next) and (j, j_next) with (i, j_next) and (j, i_next)
///     - 2-opt* (if r_i != r_j): swap (i, i_next) and (j, j_next) with (i, j) and (i_next, j_next)
///     - 2-opt* (if r_i != r_j): swap (i, i_next) and (j, j_next) with (i, j_next) and (j, i_next)
pub struct HGSLS {
  /// HGS LS internal structures
  ///
  /// Order in which to process customers
  cust_order: Vec<usize>,
  /// Shuffled nearest neighbors for each customer
  nn_order: Vec<Vec<Customer>>,
  /// Order in which to process vehicle routes
  route_order: Vec<usize>,
  /// Indices for each customer iterations; go to neighborhood search when ci == cust_order.len()
  ci: usize,
  cj: usize,
  /// Indices for each route iteration; reset LS when ri == route_order.len()
  ri: usize,
  rj: usize,
  /// Store whether an improvement was made on this iteration
  improved: bool,

  /// Current individual
  ind: Individual,
  /// Capacity penalty
  excess_penalty: f64,

  /// Pred/succs for each customer
  pred: Vec<usize>,
  succ: Vec<usize>,
  /// Routes for each customer
  cust_routes: Vec<usize>,
  /// Vehicle loads
  loads: Vec<f64>,
}

impl HGSLS {
  /// Initializes internal HGSLS structures for a given individual.
  pub fn new(ind: Individual, excess_penalty: f64) -> HGSLS {
    // Compute preds/succs/routes for each customer
    let mut pred = vec![0; ind.vrp.n_customers + 1];
    let mut succ = vec![0; ind.vrp.n_customers + 1];
    let mut cust_routes = vec![0; ind.vrp.n_customers + 1];
    // Store loads for each vehicle
    let mut loads = vec![0.; ind.vrp.n_vehicles];
    for (r, route) in ind.routes.iter().enumerate() {
      let mut load = 0.;
      for i in 0..route.len() {
        let c = route[i];
        if i == 0 {
          pred[c] = 0;
        } else {
          pred[c] = route[i-1];
        }
        if i == route.len() - 1 {
          succ[c] = 0;
        } else {
          succ[c] = route[i+1];
        }

        cust_routes[c] = r;
        load += ind.vrp.customers[c].demand;
      }
      loads[r] = load;
    }

    // Initialize HGSLS values from individual
    let mut hgsls = HGSLS {
        cust_order: ind.total_route.clone(),
        nn_order: ind.vrp.customer_nns.clone(),
        route_order: (0..ind.vrp.n_vehicles).collect(),
        ci: 0,
        cj: 0,
        ri: 0,
        rj: 0,
        improved: false,
        ind,
        excess_penalty,
        pred,
        succ,
        cust_routes,
        loads,
    };
    hgsls.reset();
    hgsls
  }

  /// (Re-)initializes the HGSLS internal structures. Assumes each one is already filled; thus, in
  /// the constructor, make sure to set values before calling this method.
  fn reset(&mut self) {
    self.ci = 0;
    self.cj = 0;
    self.ri = 0;
    self.rj = 0;
    self.improved = false;
    // Shuffle customer, route, and NN order
    self.cust_order.shuffle(&mut thread_rng());
    self.route_order.shuffle(&mut thread_rng());
    for nns in self.nn_order.iter_mut() {
      nns.shuffle(&mut thread_rng());
    }

    log::trace!("Customer order: {:?}", self.cust_order);
    // let nn_ids = self.nn_order.iter().map(|nns| nns.iter().map(|nn| nn.id).collect::<Vec<_>>()).collect::<Vec<_>>();
    // log::trace!("Customer NN order: {:?}", nn_ids);
  }

  /// If accepted, relocate i to after j
  fn relocate_i_j(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying relocate of {} after {}", i, j);

    // If j is depot or i == succ_j, skip
    if j == 0 || i == self.succ[j] {
      log::trace!("Skipping relocate; {} is depot or {} == succ j", j, i);
      return false;
    }

    // Get pred/succ for i and j
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let succ_j = self.succ[j];

    // Check distance delta for relocation:
    // - for i's route: d(pred_i, succ_i) - d(pred_i, i) - d(i, succ_i)
    // - for j's route: d(j, i) + d(i, succ_j) - d(j, succ_j)
    let dist_mtx = &self.ind.vrp.dist_mtx;
    let mut delta_i = dist_mtx[pred_i][succ_i] - dist_mtx[pred_i][i] - dist_mtx[i][succ_i];
    let mut delta_j = dist_mtx[j][i] + dist_mtx[i][succ_j] - dist_mtx[j][succ_j];

    // If swapping across routes, record change in vehicle load
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] + load_i) - penalty_j;
    }

    let net_change = delta_i + delta_j;
    if accept(net_change) {
      log::trace!("Relocating {} after {} (delta {})", i, j, net_change);

      // ugh my indexing here is ugly but oh well
      self.insert_after(j, i);

      self.ind.objective += net_change;
      return true;
    }
    false
  }

  /// If accepted, relocate (i, i_next) to after j
  fn relocate_iin_j(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying relocate ({}, {}) after {}", i, self.succ[i], j);

    log::trace!("i route: {:?}", self.ind.routes[self.cust_routes[i]]);
    log::trace!("i pred, succ, succsucc: {}, {}, {}", self.pred[i], self.succ[i], self.succ[self.succ[i]]);
    // Get pred/succ for i, i_next, and j
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let succ_succ_i = self.succ[succ_i];
    let succ_j = self.succ[j];

    // If i == succ_j, or succ_i == j, or succ_i is depot, skip
    if succ_j == i || succ_i == j || succ_i == 0 || j == 0 {
      log::trace!("Skipping relocate; {} and {} adjacent or next/j is depot", i, j);
      return false;
    }

    // Check distance delta for relocation:
    let dm = &self.ind.vrp.dist_mtx;
    let mut delta_i = dm[pred_i][succ_succ_i] - dm[pred_i][i] - dm[succ_i][succ_succ_i];
    let mut delta_j = dm[j][i] + dm[succ_i][succ_j] - dm[j][succ_j];
    // If swapping across routes, record change in vehicle load
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;
      let load_i_next = self.ind.vrp.customers[succ_i].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i - load_i_next) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] + load_i + load_i_next) - penalty_j;
    }

    let net_change = delta_i + delta_j;
    if accept(net_change) {
      log::trace!("Relocating ({}, {}) after {} (delta {})", i, self.succ[i], j, net_change);
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
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying relocate ({}, {}) after {} as ({}, {})", i, self.succ[i], j, self.succ[i], i);

    // Get pred/succ for i, i_next, and j
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let succ_i_next = self.succ[succ_i];
    let succ_j = self.succ[j];

    // If i == succ_j, or succ_i == j, or succ_i is depot, skip
    if succ_j == i || succ_i == j || succ_i == 0 || j == 0 {
      log::trace!("Skipping relocate; {} and {} adjacent or next/j is depot", i, j);
      return false;
    }

    // Check distance delta for relocation:
    // - for i's route: d(pred_i, succ_i) - d(pred_i, i) - d(i, succ_i)
    let dm = &self.ind.vrp.dist_mtx;
    let mut delta_i = dm[pred_i][succ_i_next] - dm[pred_i][i] -
      dm[i][succ_i] - dm[succ_i][succ_i_next];
    let mut delta_j = dm[j][succ_i] + dm[succ_i][i] +
      dm[i][succ_j] - dm[j][succ_j];
    // If swapping across routes, record change in vehicle load
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;
      let load_i_next = self.ind.vrp.customers[succ_i].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i - load_i_next) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] + load_i + load_i_next) - penalty_j;
    }

    let net_change = delta_i + delta_j;
    if accept(net_change) {
      log::trace!("Relocating ({}, {}) after {} as ({}, {}) (delta {})", i, self.succ[i], j, self.succ[i], i, net_change);
      self.insert_after(j, succ_i);
      self.insert_after(succ_i, i);

      self.ind.objective += net_change;
      return true;
    }
    false
  }

  /// If accepted, swap i with j
  fn swap_i_j(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying swap {} with {}", i, j);

    // Get pred/succ for i and j
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let pred_j = self.pred[j];
    let succ_j = self.succ[j];

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
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;
      let load_j = self.ind.vrp.customers[j].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i + load_j) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] - load_j + load_i) - penalty_j;
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
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying swap ({}, {}) with {}", i, self.succ[i], j);

    // Get pred/succ for i, i_next, and j
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let succ_succ_i = self.succ[succ_i];
    let pred_j = self.pred[j];
    let succ_j = self.succ[j];

    // If i == pred_j or i_next == pred_j or i == succ_j or succ_i == depot, skip
    if i == pred_j || succ_i == pred_j || i == succ_j || succ_i == 0 {
      log::trace!("Skipping swap; {} and {} adjacent or next is depot", i, j);
      return false;
    }

    // Compute distance delta for swap:
    let dm = &self.ind.vrp.dist_mtx;
    let mut delta_i = dm[pred_i][j] + dm[j][succ_succ_i]
      - dm[pred_i][i] - dm[succ_i][succ_succ_i];
    let mut delta_j = dm[pred_j][i] + dm[succ_i][succ_j]
      - dm[pred_j][j] - dm[j][succ_j];

    // If swapping across routes, record change in vehicle load
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;
      let load_i_next = self.ind.vrp.customers[succ_i].demand;
      let load_j = self.ind.vrp.customers[j].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i - load_i_next + load_j) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] - load_j + load_i + load_i_next) - penalty_j;
    }

    let net_change = delta_i + delta_j;
    if accept(net_change) {
      log::trace!("Swapping ({}, {}) with {} (delta {})", i, self.succ[i], j, net_change);
      self.swap(i, j);
      self.insert_after(i, succ_i);

      self.ind.objective += net_change;
      return true;
    }
    false
  }

  /// If accepted, swap (i, i_next) with (j, j_next)
  fn swap_iin_jjn(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying swap ({}, {}) with ({}, {})", i, self.succ[i], j, self.succ[j]);

    // Get pred/succ for i, i_next, j, and j_next
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let succ_succ_i = self.succ[succ_i];
    let pred_j = self.pred[j];
    let succ_j = self.succ[j];
    let succ_succ_j = self.succ[succ_j];

    // If succ_i or succ_j is depot, or succ_j == pred_i or i == succ_j or succ_i == j or j == succ_succ_i, skip
    if succ_i == 0 || succ_j == 0 || succ_j == pred_i || i == succ_j || succ_i == j || j == succ_succ_i {
      log::trace!("Skipping swap; {} and {} adjacent or next is depot", i, j);
      return false;
    }

    // Compute distance delta for swap:
    let dm = &self.ind.vrp.dist_mtx;
    let mut delta_i = dm[pred_i][j] + dm[succ_j][succ_succ_i]
      - dm[pred_i][i] - dm[succ_i][succ_succ_i];
    let mut delta_j = dm[pred_j][i] + dm[succ_i][succ_succ_j]
      - dm[pred_j][j] - dm[succ_j][succ_succ_j];

    // If swapping across routes, record change in vehicle load
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      let load_i = self.ind.vrp.customers[i].demand;
      let load_i_next = self.ind.vrp.customers[succ_i].demand;
      let load_j = self.ind.vrp.customers[j].demand;
      let load_j_next = self.ind.vrp.customers[succ_j].demand;

      // See if load exceeds capacity in either swap
      let penalty_i = self.load_penalty(self.loads[ri]);
      let penalty_j = self.load_penalty(self.loads[rj]);
      delta_i += self.load_penalty(self.loads[ri] - load_i - load_i_next + load_j + load_j_next) - penalty_i;
      delta_j += self.load_penalty(self.loads[rj] - load_j - load_j_next + load_i + load_i_next) - penalty_j;
    }

    let net_change = delta_i + delta_j;
    if accept(net_change) {
      log::trace!("Swapping ({}, {}) with ({}, {}) (delta {})", i, self.succ[i], j, self.succ[j], net_change);
      self.swap(i, j);
      self.swap(succ_i, succ_j);

      self.ind.objective += net_change;
      return true;
    }
    false
  }

  /// If accepted, apply 2-opt move
  fn two_opt(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying 2-opt move on {} and {}", i, j);

    // Only apply if on same route
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri != rj {
      log::trace!("Skipping 2-opt move; {} and {} on different routes", i, j);
      return false;
    }
    // If less than 3 customers, skip
    let route = &mut self.ind.routes[ri];
    if route.len() < 3 {
      log::trace!("Skipping 2-opt move; route {} has less than 3 customers", ri);
      return false;
    }

    // Get succ for i and j
    let succ_i = self.succ[i];
    let succ_j = self.succ[j];

    // Compute distance delta for 2-opt swap:
    let dm = &self.ind.vrp.dist_mtx;
    let delta = dm[i][j] + dm[succ_i][succ_j] - dm[i][succ_i] - dm[j][succ_j];

    if accept(delta) {
      log::trace!("Applying 2-opt move on {} and {} (delta {})", i, j, delta);

      // Find indices of i and j (reverse =j)
      let idx_i = route.iter().position(|&x| x == i).unwrap();
      let idx_j = route.iter().position(|&x| x == j).unwrap();
      // Sort
      let (idx_i, idx_j) = if idx_i < idx_j { (idx_i, idx_j) } else { (idx_j, idx_i) };

      // Update preds/succs, since each customer will be reversed
      for k in (idx_i+1)..idx_j {
        let c = route[k];
        let pred_c = route[k-1];
        let succ_c = route[k+1];
        self.pred[c] = succ_c;
        self.succ[c] = pred_c;
      }

      // Specially preds/succs of i, j, succ_i, and succ_j
      self.succ[i] = j;
      self.pred[j] = i;
      self.succ[succ_i] = succ_j;
      self.pred[succ_j] = succ_i;

      // Reverse the segment between i and =j
      route[(idx_i+1)..=idx_j].reverse();

      self.ind.objective += delta;
      return true;
    }
    false
  }

  /// If accepted, apply 2-opt* move
  fn two_opt_star(&mut self, accept: &AcceptOperator) -> bool {
    let i = self.cust_order[self.ci];
    let j = self.nn_order[i][self.cj].id;

    log::trace!("Trying 2-opt* move on {} and {}", i, j);

    // Get succ for i and j
    let succ_i = self.succ[i];
    let succ_j = self.succ[j];

    // Only apply if on different routes
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri == rj {
      log::trace!("Skipping 2-opt* move; {} and {} on same route", i, j);
      return false;
    }

    // Compute distance delta for 2-opt* swap
    let dm = &self.ind.vrp.dist_mtx;
    let mut delta = dm[i][succ_j] + dm[j][succ_i] - dm[i][succ_i] - dm[j][succ_j];

    let mut route_i = self.ind.routes[ri].clone();
    let mut route_j = self.ind.routes[rj].clone();
    let idx_i = route_i.iter().position(|&x| x == i).unwrap();
    let idx_j = route_j.iter().position(|&x| x == j).unwrap();

    // Compute change in load: all customers after i are moved to j's route, and vice versa
    let mut moved_load_ri = 0.;
    for k in route_i[(idx_i+1)..].iter() {
      moved_load_ri += self.ind.vrp.customers[*k].demand;
    }
    let mut moved_load_rj = 0.;
    for k in route_j[(idx_j+1)..].iter() {
      moved_load_rj += self.ind.vrp.customers[*k].demand;
    }

    let penalty_i = self.load_penalty(self.loads[ri]);
    let penalty_j = self.load_penalty(self.loads[rj]);
    delta += self.load_penalty(self.loads[ri] - moved_load_ri + moved_load_rj) - penalty_i;
    delta += self.load_penalty(self.loads[rj] - moved_load_rj + moved_load_ri) - penalty_j;

    if accept(delta) {
      log::trace!("Applying 2-opt* move on {} and {} (delta {})", i, j, delta);

      // If accept, then move [succ_j:] to after i in route i, and [succ_i:] to after j in route j

      // Drain idx_i+1.. from route i, and idx_j+1 from route j
      let after_i = route_i.drain((idx_i+1)..).collect::<Vec<_>>();
      let after_j = route_j.drain((idx_j+1)..).collect::<Vec<_>>();
      route_i.extend(after_j);
      route_j.extend(after_i);

      // Re-compute preds/succs/routes/loads for the each customer in the two routes
      self.loads[ri] = 0.;
      for i in 0..route_i.len() {
        let c = route_i[i];
        if i == 0 {
          self.pred[c] = 0;
        } else {
          self.pred[c] = route_i[i-1];
        }
        if i == route_i.len() - 1 {
          self.succ[c] = 0;
        } else {
          self.succ[c] = route_i[i+1];
        }
        self.cust_routes[c] = ri;
        self.loads[ri] += self.ind.vrp.customers[c].demand;
      }
      self.loads[rj] = 0.;
      for j in 0..route_j.len() {
        let c = route_j[j];
        if j == 0 {
          self.pred[c] = 0;
        } else {
          self.pred[c] = route_j[j-1];
        }
        if j == route_j.len() - 1 {
          self.succ[c] = 0;
        } else {
          self.succ[c] = route_j[j+1];
        }
        self.cust_routes[c] = rj;
        self.loads[rj] += self.ind.vrp.customers[c].demand;
      }

      log::trace!("before swapping routes: {:?}, {:?}", self.ind.routes[ri], self.ind.routes[rj]);
      self.ind.routes[ri] = route_i;
      self.ind.routes[rj] = route_j;

      self.ind.objective += delta;

      log::trace!("after swapping routes: {:?}, {:?}", self.ind.routes[ri], self.ind.routes[rj]);
      log::trace!("preds: {:?}", self.pred);
      log::trace!("succs: {:?}", self.succ);
      return true;
    }

    false
  }

  /// If accepted, apply 2-opt* move with flipped order
  fn two_opt_star_flip(&mut self, accept: &AcceptOperator) -> bool {
    // TODO: actually implement
    return false;
  }

  /// Attempt to apply swap* operator, swapping routes based on polar sectors.
  fn swap_star(&mut self, accept: &AcceptOperator) -> bool {
    // TODO: actually implement
    return false;
  }

  /// Get the customer id associated with the customer index
  fn get_cust_id(&self, i: usize) -> usize {
    self.cust_order[i]
  }

  /// Get the NN associated with the customer index
  fn get_nn_id(&self, i: usize, j: usize) -> usize {
    self.nn_order[i][j].id
  }

  /// Get the route id associated with the route index
  fn get_route_id(&self, r: usize) -> usize {
    self.route_order[r]
  }

  /// Computes the excess penalty for a load
  fn load_penalty(&self, load: f64) -> f64 {
    (load - self.ind.vrp.vehicle_cap as f64).max(0.) * self.excess_penalty
  }

  /// Inserts customer w/ id j after cstomer w/ id i
  fn insert_after(&mut self, i: usize, j: usize) {
    let succ_i = self.succ[i];
    let pred_j = self.pred[j];
    let succ_j = self.succ[j];

    // Update pred/succ for i and j
    self.succ[i] = j;
    self.pred[succ_i] = j;
    self.succ[j] = succ_i;
    self.pred[j] = i;
    self.succ[pred_j] = succ_j;
    self.pred[succ_j] = pred_j;

    // Remove j from its route
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
    if ri == rj {
      let route = &mut self.ind.routes[rj];
      let idx = route.iter().position(|&x| x == j).unwrap();
      log::trace!("Moving {} to after {} in route {:?}", j, i, route);
      route.remove(idx);
      // Insert j after i in i's route
      let idx = route.iter().position(|&x| x == i).unwrap();
      route.insert(idx + 1, j);
    } else {
      // Update loads for both routes
      let load_j = self.ind.vrp.customers[j].demand;
      self.loads[rj] -= load_j;
      self.loads[ri] += load_j;
      // Change routes of j
      self.cust_routes[j] = ri;

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

    log::trace!("After insert, ri: {:?}, rj: {:?}", self.ind.routes[ri], self.ind.routes[rj]);

    log::trace!("{i} pred, succ: {}, {}", self.pred[i], self.succ[i]);
    log::trace!("{j} pred, succ: {}, {}", self.pred[j], self.succ[j]);
    log::trace!("{succ_i} pred, succ: {}, {}", self.pred[succ_i], self.succ[succ_i]);
    log::trace!("{pred_j} pred, succ: {}, {}", self.pred[pred_j], self.succ[pred_j]);
  }

  /// Swaps customer w/ id i with customer w/ id j
  fn swap(&mut self, i: usize, j: usize) {
    let pred_i = self.pred[i];
    let succ_i = self.succ[i];
    let pred_j = self.pred[j];
    let succ_j = self.succ[j];

    // Update pred/succ for i and j
    self.succ[pred_i] = j;
    self.pred[succ_i] = j;
    self.succ[pred_j] = i;
    self.pred[succ_j] = i;
    self.pred[i] = pred_j;
    self.succ[i] = succ_j;
    self.pred[j] = pred_i;
    self.succ[j] = succ_i;

    // Update routes for i and j
    let ri = self.cust_routes[i];
    let rj = self.cust_routes[j];
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
      self.loads[ri] += load_j - load_i;
      self.loads[rj] += load_i - load_j;
      // Change routes of i and j
      self.cust_routes[i] = rj;
      self.cust_routes[j] = ri;

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

impl MHOperator for HGSLS {
  fn ls_move(&mut self, accept: &AcceptOperator) -> bool {
    loop {
      log::trace!("here: ci: {}, cj: {}, ri: {}, rj: {} (cust_order len: {}, route_order len: {})", self.ci, self.cj, self.ri, self.rj, self.cust_order.len(), self.route_order.len());
      // If still have customer to do, process
      if self.ci < self.cust_order.len() {
        let ci = self.cust_order[self.ci];
        // If still have NN to do, process by sequentially trying any of the operators
        if self.cj < self.nn_order[ci].len() {
          log::trace!("Processing customer num {}'s NN num {} (ci: {}, cj: {})", self.cust_order[self.ci], self.nn_order[self.cust_order[self.ci]][self.cj].id, self.ci, self.cj);

          // // Get IDs from NNs
          // let i = self.cust_order[self.ci];
          // let i_nns = self.nn_order[i].iter().map(|nn| nn.id).collect::<Vec<_>>();
          // // Check if i_nns contains i
          // if i_nns.contains(&i) {
          //   log::error!("NNs for customer {} contain itself", i);
          // }
          // log::trace!("Customer {} nns: {:?}", i, i_nns);

          let mut improved = false;
          if self.relocate_i_j(accept) {
            improved = true;
          } else if self.relocate_iin_j(accept) {
            improved = true;
          } else if self.relocate_iin_j_flip(accept) {
            improved = true;
          } else if self.swap_i_j(accept) {
            improved = true;
          } else if self.swap_iin_j(accept) {
            improved = true;
          } else if self.swap_iin_jjn(accept) {
            improved = true;
          } else if self.two_opt(accept) {
            improved = true;
          } else if self.two_opt_star(accept) {
            improved = true;
          }

          self.cj += 1;
          if improved {
            self.improved = true;
            return true;
          }

          // If no success, continue loop to next NN
          log::trace!("No success, continuing to next NN");
        } else {
          // If processed all NNs, go to next customer and reset cj
          self.ci += 1;
          self.cj = 0;
        }
      } else {
        // Otherwise, if still have route to do, process
        if self.ri < self.route_order.len() {
          // If have another route to do, process with swap*
          if self.rj < self.route_order.len() {
            // If ri == rj, continue
            if true || self.ri == self.rj {
              self.rj += 1;
              continue;
            }

            // If either route is empty, skip
            let ri = self.route_order[self.ri];
            let rj = self.route_order[self.rj];
            if self.ind.routes[ri].len() == 0 || self.ind.routes[rj].len() == 0 {
              self.rj += 1;
              continue;
            }

            log::trace!("Processing swap* for route nums {} and {}", self.get_route_id(self.ri), self.get_route_id(self.rj));
            log::trace!("Route {} customers: {:?}", self.get_route_id(self.ri), self.ind.routes[self.route_order[self.ri]]);
            log::trace!("Route {} customers: {:?}", self.get_route_id(self.rj), self.ind.routes[self.route_order[self.rj]]);

            // Check if polar sectors overlap
            let polars_i = self.ind.routes[self.route_order[self.ri]].iter().map(|&c| self.ind.vrp.customers[c].polar_angle).collect::<Vec<_>>();
            let polars_j = self.ind.routes[self.route_order[self.rj]].iter().map(|&c| self.ind.vrp.customers[c].polar_angle).collect::<Vec<_>>();
            let psi = PolarSector::from_points(&polars_i);
            let psj = PolarSector::from_points(&polars_j);
            if psi.overlaps(&psj) {
              if self.swap_star(accept) {
                self.improved = true;
                self.rj += 1;
                return true;
              }
            }
            self.rj += 1;
          } else {
            // Otherwise, go to next route and reset rj
            self.ri += 1;
            self.rj = 0;
          }
        } else {
          // If no improvement, reset and return false
          if !self.improved {
            self.reset();
            return false;
          } else {
            // Reset to beginning
            log::debug!("Resetting HGSLS w/ new random order");
            self.reset();
            return true;
          }
        }
      }
    }
  }

  fn get_individual(&self) -> Individual {
    // Reconstruct total route from individual routes
    let mut total_route: Vec<_> = vec![];
    for route in self.ind.routes.iter() {
      total_route.extend(route.iter());
    }
    let mut new_ind = self.ind.clone();
    new_ind.total_route = total_route;
    new_ind
  }
}