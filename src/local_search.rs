use std::{time::{Duration, Instant}};

use rand::random;

use crate::{ls_operators::{MHOperator, HGSLS}, Individual};

const EPSILON: f64 = 0.00001;

/// Struct implementing local search algorithms: given an individual with some (potentially
/// infeasible) solution, attempt to find better solutions.
pub struct LocalSearch {
  /// Store best and current individual so far
  best: Individual,
  curr: Individual,
}

impl LocalSearch {
  /// Create a new local search instance from an individual.
  pub fn new(ind: Individual) -> LocalSearch {
    LocalSearch { best: ind.clone(), curr: ind }
  }

  /// Runs local search with the specified parameters, returning the best solution found.
  /// - `accept_temp`: temperature for SA to accept worse solution
  /// - `time_limit`: time limit for search
  /// - `ni_limit`: number of iterations without improvement until termination
  pub fn run(&mut self, accept_temp: f64, time_limit: Duration, ni_limit: usize, excess_penalty: f64) -> Individual {
    let mut hgs = HGSLS::new(self.curr.clone(), excess_penalty);

    let accept = sa_accept(accept_temp);

    let start = Instant::now();
    // Record number of moves without improvement
    let mut ni = 0;
    // Loop until time limit or no improvement
    'ls: loop {
      // If time limit exceeded, break
      if start.elapsed() >= time_limit {
        log::debug!("Time limit exceeded");
        break 'ls;
      }
      // If no improvement limit exceeded, break
      if ni >= ni_limit {
        log::debug!("No improvement limit exceeded");
        break 'ls;
      }

      // Performs a LS move to get a new solution
      let res = hgs.ls_move(&accept);
      // If succeeded, get the individual
      if res {
        let mut new_ind = hgs.get_individual();
        new_ind.objective();
        // If new solution is better (i.e. lower objective), update current
        if new_ind.objective <= self.curr.objective {
          log::debug!("Updating curr to new solution (old {}, new {})", self.curr.objective, new_ind.objective);
          self.curr = new_ind;
          // If new solution is best, update best
          if self.curr.objective <= self.best.objective {
            log::debug!("Got better solution (old {}, new {})", self.best.objective, self.curr.objective);
            log::debug!("Old solution dist, cap: {}, {}", self.best.total_dist, self.best.excess_cap);
            log::debug!("New solution dist, cap: {}, {}", self.curr.total_dist, self.curr.excess_cap);
            self.best = self.curr.clone();
            // Reset number of moves without improvement
            ni = 0;
          }
        }
      } else {
        // If no improvement, break
        log::debug!("No improvement; stopping LS");
        break 'ls;
        // TODO: check if we should keep looping
        // If new solution is worse, increment number of moves without improvement
        ni += 1;
      }
    }

    // Return the best solution found
    self.best.clone()
  }

  fn ls_move(&mut self) -> Individual {
    // TODO: call ind.objective() on new individuals
    todo!()
  }
}

/// Simulated annealing accept fn using temp.
fn sa_accept(temp: f64) -> Box<dyn Fn(f64) -> bool> {
  Box::new(move |delta: f64| -> bool {
    // If delta negative, it lowers objective value, so accept
    if delta < -EPSILON {
      return true;
    } else {
      return false;
    }
    // Otherwise, accept with probability based on epsilon
    // log::trace!("accept probability: {}", (delta / temp).exp());
    // random::<f64>() < (delta / temp).exp()
  })
}

/// Performs semi-optimal two-opt: randomly selects a vehicle route, then finds the best two-opt
/// move.
fn two_opt(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 3 nodes, return
  if len < 3 {
    return ind;
  }

  let dist_mtx = &ind.vrp.dist_mtx;
  // Find best two-opt move
  let mut best_delta = 0.0;
  let (mut best_i, mut best_j) = (0, 0);
  // len-2 here, since we need to consider two nodes at a time
  for i in 0..(len-2) {
    // len-1 here, since the "endpoint" of this node is the last node
    for j in (i+1)..(len-1) {
      // Compute the distance delta of the two-opt move, and see if better
      // (i, i+1) and (j, j+1) are the two edges being swapped
      let delta = dist_mtx[route[i]][route[j]] + dist_mtx[route[i+1]][route[j+1]]
        - dist_mtx[route[i]][route[i+1]] - dist_mtx[route[j]][route[j+1]];
      // If this swap is better, update best delta and indices
      if delta < best_delta {
        best_delta = delta;
        best_i = i;
        best_j = j;
      }
    }
  }

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Reverse the path between i and j
  route[best_i..=best_j].reverse();

  new_ind
}

/// Performs random two-opt: randomly selects a vehicle route, then randomly selects a segment to
/// reverse.
fn random_two_opt(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 3 nodes, return
  if len < 3 {
    return ind;
  }

  // Randomly select segment length + start to two-opt
  let segment_len = 2 + random::<usize>() % (len-2);
  let start = random::<usize>() % (len-segment_len);
  let end = start + segment_len;

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Reverse the segment TODO: ..end vs ..=end?
  route[start..=end].reverse();

  new_ind
}

/// Performs random three-opt: randomly selects a vehicle route, then randomly selects segments to
/// swap if it performs better.
fn random_three_opt(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 6 nodes, return
  if len < 6 {
    return ind;
  }

  // Credit to Zachary Espiritu for the indexing that I couldn't think of lol

  // Select 6 random indices
  let a = random::<usize>() % (len-6);
  let b = a + random::<usize>() % (len-a-5);
  let c = b + random::<usize>() % (len-b-4);
  let d = c + random::<usize>() % (len-c-3);
  let e = d + random::<usize>() % (len-d-2);
  let f = e + random::<usize>() % (len-e-1);

  let (ra, rb, rc, rd, re, rf) = (route[a], route[b], route[c], route[d], route[e], route[f]);
  // Check which reversal performs best
  let dist_mtx = &ind.vrp.dist_mtx;
  let d0 = dist_mtx[ra][rb] + dist_mtx[rc][rd] + dist_mtx[re][rf];
  let d1 = dist_mtx[ra][rc] + dist_mtx[rb][rd] + dist_mtx[re][rf];
  let d2 = dist_mtx[ra][rb] + dist_mtx[rc][re] + dist_mtx[rd][rf];
  let d3 = dist_mtx[ra][rd] + dist_mtx[re][rb] + dist_mtx[rc][rf];
  let d4 = dist_mtx[rb][rf] + dist_mtx[rc][rd] + dist_mtx[re][ra];

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Perform the best reversal
  if d1 < d0 {
    route[b..d].reverse();
  } else if d2 < d0 {
    route[d..f].reverse();
  } else if d3 < d0 {
    // Swap b..d and d..f
    let tmp = route[b..f].to_vec();
    for i in d..f {
      route[b+i] = tmp[i-b];
    }
    for i in b..d {
      route[b+(f-d)+i] = tmp[i-b];
    }
  } else if d4 < d0 {
    route[b..f].reverse();
  }

  new_ind
}

/// Performs semi-optimal exchange: randomly selects a route, and finds the best exchange move for
/// two nodes.
fn exchange(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 2 nodes, return
  if len < 2 {
    return ind;
  }

  // Find best exchange move
  let mut best_delta = 0.0;
  let (mut best_i, mut best_j) = (0, 0);

  let dist_mtx = &ind.vrp.dist_mtx;
  for i in 0..(len-1) {
    for j in (i+1)..len {
      let (ri, rj) = (route[i], route[j]);
      // Get previous elements in route; if at start, go to depot
      let ri_prev = if i == 0 { 0 } else { route[i-1] };
      let rj_prev = route[j-1];
      // Get next elements in route; if at end, go to depot
      let ri_next = route[i+1];
      let rj_next = if j == len-1 { 0 } else { route[j+1] };

      // Compute the distance delta of the exchange move, and see if better:
      let delta = dist_mtx[ri_prev][rj] + dist_mtx[rj][ri_next]
        + dist_mtx[rj_prev][ri] + dist_mtx[ri][rj_next]
        - dist_mtx[ri_prev][ri] - dist_mtx[ri][ri_next]
        - dist_mtx[rj_prev][rj] - dist_mtx[rj][rj_next];
      // If this swap is better, update best delta and indices
      if delta < best_delta {
        best_delta = delta;
        best_i = i;
        best_j = j;
      }
    }
  }

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Swap the two nodes
  route.swap(best_i, best_j);

  new_ind
}

fn random_exchange(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 2 nodes, return
  if len < 2 {
    return ind;
  }

  // Randomly select two nodes to swap
  let i = random::<usize>() % len;
  let j = random::<usize>() % len;

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Swap the two nodes
  route.swap(i, j);

  new_ind
}

/// Performs semi-optimal relocation: randomly selects a route, and finds the best relocation move
fn relocate(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 2 nodes, return
  if len < 2 {
    return ind;
  }

  let dist_mtx = &ind.vrp.dist_mtx;
  // Find best relocation move (i -> j)
  let mut best_delta = 0.0;
  let (mut best_i, mut best_j) = (0, 0);
  for i in 0..len {
    for j in 0..len {
      let (ri, rj) = (route[i], route[j]);
      // Get previous elements in route; if at start, go to depot
      let ri_prev = if i == 0 { 0 } else { route[i-1] };
      let rj_prev = if j == 0 { 0 } else { route[j-1] };
      // Get next elements in route; if at end, go to depot
      let ri_next = if i == len-1 { 0 } else { route[i+1] };
      let rj_next = if j == len-1 { 0 } else { route[j+1] };

      let delta = dist_mtx[ri_prev][rj] + dist_mtx[rj][ri_next]
        - dist_mtx[ri_prev][ri] - dist_mtx[ri][ri_next]
        + dist_mtx[rj_prev][ri] + dist_mtx[ri][rj_next]
        - dist_mtx[rj_prev][rj] - dist_mtx[rj][rj_next];

      // If this swap is better, update best delta and indices
      if delta < best_delta {
        best_delta = delta;
        best_i = i;
        best_j = j;
      }
    }
  }

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Relocate node i to position j
  let node = route.remove(best_i);
  // If best_i < best_j, then best_j is shifted left by 1
  let best_j = if best_i < best_j { best_j - 1 } else { best_j };
  route.insert(best_j, node);

  new_ind
}

/// Performs random relocation: randomly selects a route, and randomly selects a node to relocate
fn random_relocate(ind: Individual) -> Individual {
  // Randomly select a vehicle route
  let route_idx = random::<usize>() % ind.routes.len();
  let route = &ind.routes[route_idx];
  let len = route.len();
  // If route is less than 2 nodes, return
  if len < 2 {
    return ind;
  }

  // Randomly select two nodes to swap
  let i = random::<usize>() % len;
  let j = random::<usize>() % len;

  let mut new_ind = ind.clone();
  let route = &mut new_ind.routes[route_idx];
  // Relocate node i to position j
  let node = route.remove(i);
  // If i < j, then j is shifted left by 1
  let j = if i < j { j - 1 } else { j };
  route.insert(j, node);

  new_ind
}


/*

intra-route:
- 2-opt: (randomly|optimally) select two nodes and reverse path in-between
- 3-opt: (randomly|optimally) select three nodes and reverse path in-between each
- exchange: (randomly|optimally) select two nodes and swap them
- relocate: (randomly|optimally) select a node and move it to a new position

inter-route: (maybe only consider inter-routes with overlapping sectors?? otherwise might not
  be worth it idk)
- 2-opt*: (randomly|optimally) select two nodes from different routes and swap their ends
- insert: (randomly|optimally) select a node from one route and insert it into another
- swap*: select two nodes, and find best insertion in each

route re-construction:
- greedy nn

Simulated Annealing:
- start with higher epsilon for worst solutions and gradually decrease
*/