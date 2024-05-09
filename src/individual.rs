use std::{
    collections::{BinaryHeap, HashSet},
    fs,
    io::Write,
    sync::Arc,
};

use ordered_float::OrderedFloat;
use rand::{seq::SliceRandom, thread_rng};

use crate::vrp_instance::VRPInstance;

type F64 = OrderedFloat<f64>;

// Represent INF using some really big number
const INF: f64 = 1e30;
const SMALLER_THAN_INF: f64 = 1e25;
/// Represents an individual solution within the population.
#[derive(Debug, Clone)]
pub struct Individual {
    /// Total route for all vehicles
    pub total_route: Vec<usize>,
    /// Vehicle routes
    pub routes: Vec<Vec<usize>>,

    /// Successor/predecessors for each customer
    pub succ: Vec<usize>,
    pub pred: Vec<usize>,
    /// Routes for each customer
    pub cust_routes: Vec<usize>,
    /// Vehicle loads
    pub loads: Vec<f64>,

    /// Fitness of this individual
    pub fitness: f64,

    /// Individuals in close proximity (for diversity calculation)
    /// TODO: how to determine "proximity"?
    // pub close_inds: BinaryHeap<>

    /// Stats for this individual
    ///
    /// Vehicles in use
    pub vehicles_used: usize,
    /// Total distance travelled
    pub total_dist: f64,
    /// Excess capacity used across all vehicles
    pub excess_cap: usize,
    /// Objective value
    pub objective: f64,

    /// VRP instance
    pub vrp: Arc<VRPInstance>,
}

impl Individual {
    /// Initializes an individual from the VRP instance.
    pub fn new(vrp: Arc<VRPInstance>) -> Individual {
        let routes = vec![vec![]; vrp.n_vehicles];

        // Randomly initialize total route
        let mut total_route = Vec::with_capacity(vrp.n_customers);
        for i in 0..vrp.n_customers {
            total_route.push(i + 1);
        }
        total_route.shuffle(&mut thread_rng());

        // Compute pred/succ/cust_routes/loads for each customer/route
        let pred = vec![0; vrp.n_customers + 1];
        let succ = vec![0; vrp.n_customers + 1];
        let cust_routes = vec![0; vrp.n_customers + 1];
        let loads = vec![0.0; vrp.n_customers + 1];

        Individual {
            total_route,
            routes,
            pred,
            succ,
            cust_routes,
            loads,
            fitness: 0.0,
            vehicles_used: 0,
            total_dist: 0.0,
            excess_cap: 0,
            objective: 0.0,
            vrp,
        }
    }

    /// Compute objective as distance traveled, plus a penalty for excess
    /// capacity
    pub fn objective(&mut self) -> f64 {
        // Compute total distance and excess capacity used
        self.total_dist = 0.;
        self.excess_cap = 0;
        for route in self.routes.iter() {
            let mut load = 0.;
            let mut dist = 0.0;
            for (i, &c) in route.iter().enumerate() {
                // If 0 or last customer, use depot, then customer
                if i == 0 {
                    dist += self.vrp.dist_mtx[0][c];
                } else if i == route.len() - 1 {
                    dist += self.vrp.dist_mtx[c][0];
                } else {
                    dist += self.vrp.dist_mtx[route[i - 1]][c];
                }
                load += self.vrp.customers[c].demand;
            }
            self.total_dist += dist;
            self.excess_cap += (load as i64 - self.vrp.vehicle_cap as i64).max(0) as usize;
        }
        self.objective =
            self.total_dist + (self.excess_cap as f64 * self.vrp.params.excess_penalty);
        log::debug!(
            "Total distance: {:.3}, excess capacity: {}, objective: {:.3}",
            self.total_dist,
            self.excess_cap,
            self.objective
        );
        self.objective
    }

    /// Feasible only if no excess capacity
    pub fn is_feasible(&self) -> bool {
        self.excess_cap == 0
    }

    /// Converts this individual back to a solution
    pub fn save_solution_default(&mut self) {
        let file_name = format!("solutions/{}.sol", self.vrp.instance_name);
        log::info!("Solution file: {}", file_name);
        self.save_solution(&file_name);
    }

    pub fn save_solution(&mut self, file_name: &str) {
        let mut f = fs::File::create(file_name).unwrap();
        let obj = self.objective();
        // Write objective to file
        writeln!(f, "{}", obj).unwrap();
        // For each vehicle, write its route if non-empty
        for route in self.routes.iter() {
            log::debug!("Route: {:?}", route);
            if route.is_empty() {
                continue;
            }
            // Write route
            for (i, c) in route.iter().enumerate() {
                // If 0 or last customer, write depot, then customer
                if i == 0 {
                    write!(f, "0 {} ", c).unwrap();
                } else if i == route.len() - 1 {
                    write!(f, "{} 0", c).unwrap();
                } else {
                    write!(f, "{} ", c).unwrap();
                }
            }
            writeln!(f).unwrap();
        }
    }

    /// Initializes metadata surrounding the individual (i.e. pred, succ,
    /// cust_routes, loads, objective)
    pub fn initialize_metadata(&mut self) {
        for (r, route) in self.routes.iter().enumerate() {
            let mut load = 0.;
            for i in 0..route.len() {
                let c = route[i];
                self.pred[c] = if i == 0 { 0 } else { route[i - 1] };
                self.succ[c] = if i == route.len() - 1 {
                    0
                } else {
                    route[i + 1]
                };
                self.cust_routes[c] = r;
                load += self.vrp.customers[c].demand as f64;
            }
            self.loads[r] = load;
        }

        // Re-build total route
        // Set objective
        self.objective();
    }

    /// Splits the total route into vehicle routes using Bellman's algorithm in
    /// topological order. Used not only in feasible solution instantiation,
    /// but also in genetic crossover.
    pub fn bellman_split(&mut self) -> bool {
        log::info!("Current tour: {:?}", self.total_route);

        let vrp = &self.vrp;
        // Predecessors for each customer in each vehicle
        let mut pred = vec![vec![0; vrp.n_customers + 1]; vrp.n_vehicles + 1];
        // DP for lowest cost from depot to customer (j) using vehicle (i)
        let mut cost_to_c = vec![vec![INF; vrp.n_customers + 1]; vrp.n_vehicles + 1];
        // Set cost to depot to 0 for all vehicles
        for v in 0..(vrp.n_vehicles + 1) {
            cost_to_c[v][0] = 0.0;
        }

        // For each vehicle, compute its lowest costs to each customer
        for v in 0..vrp.n_vehicles {
            // For each customer, compute the lowest cost to reach it using this vehicle
            for c in v..vrp.n_customers {
                // Only attempt if less than "infinity"
                if cost_to_c[v][c] < INF {
                    log::trace!("Computing cost for vehicle {} to customer {}", v, c);

                    let mut load = 0.;
                    let mut dist = 0.;

                    // For each successive customer, add its distance and demand
                    for next in (c + 1)..(vrp.n_customers + 1) {
                        // If load is too high, stop considering
                        if load >= 1.5 * vrp.vehicle_cap as f64 {
                            log::trace!(
                                "Load {} too high for vehicle {}; stopping at {}",
                                load,
                                v,
                                next
                            );
                            break;
                        }

                        // Get the actual customer on the tour; -1 to satisfy indexing in DP
                        let cust_next = self.total_route[next - 1];

                        load += vrp.customers[cust_next].demand as f64;
                        // If next is c+1, use distance from depot; otherwise, use distance from
                        // previous customer
                        if next == c + 1 {
                            dist += vrp.dist_mtx[0][cust_next];
                        } else {
                            // Get prev customer
                            let cust_prev = self.total_route[next - 2];
                            dist += vrp.dist_mtx[cust_prev][cust_next];
                        }
                        // Compute cost using current distance + distance from here back to depot +
                        // a penalty if load is too high
                        let cost = dist
                            + vrp.dist_mtx[next][0]
                            + vrp.params.excess_penalty * (load - vrp.vehicle_cap as f64).max(0.);
                        dist = vrp.dist_mtx[c][next];
                        if cost_to_c[v][c] + cost < cost_to_c[v + 1][next] {
                            cost_to_c[v + 1][next] = cost_to_c[v][c] + cost;
                            pred[v + 1][next] = c;
                        }
                    }
                }
            }
        }

        // Check if it managed to reach last node
        if cost_to_c[vrp.n_vehicles][vrp.n_customers] > SMALLER_THAN_INF {
            log::warn!("Bellman split failed: no route to last customer");
            return false;
        }
        log::trace!("cost[veh][cust]: {:?}", cost_to_c);
        // Reconstruct the routes
        let mut end = vrp.n_customers;
        for v in (0..vrp.n_vehicles).rev() {
            self.routes[v].clear();
            let start = pred[v + 1][end];
            for next in start..end {
                self.routes[v].push(self.total_route[next]);
            }
            end = start;
        }

        // Check if it cycled back to beginning (i.e. depot)
        if end != 0 {
            log::warn!("Bellman split failed to cycle back to depot");
            return false;
        }

        // Set metadata
        self.initialize_metadata();
        true
    }

    /// Initializes a (potentially infeasible) solution using parallel greedy NN
    /// search.
    pub fn greedy_nn(&mut self) -> bool {
        // record assigned customers
        let mut seen = HashSet::new();

        // Take first n_vehicles customers as starting points
        for v in 0..self.vrp.n_vehicles {
            self.routes[v].push(self.total_route[v]);
            seen.insert(self.total_route[v]);
        }

        // For each vehicle, find nearest neighbor until all assigned
        'nn: loop {
            for v in 0..self.vrp.n_vehicles {
                if seen.len() == self.vrp.n_customers {
                    break 'nn;
                }
                let last = self.routes[v].last().unwrap();

                // Don't use kd tree here, since need to keep track of seen customers
                let mut best = 0;
                let mut best_dist = INF;
                for c in 1..(self.vrp.n_customers + 1) {
                    if seen.contains(&c) {
                        continue;
                    }
                    let dist = self.vrp.dist_mtx[*last][c];
                    if dist < best_dist {
                        best = c;
                        best_dist = dist;
                    }
                }
                self.routes[v].push(best);
                seen.insert(best);
            }
        }

        // Initialize metadata
        self.initialize_metadata();
        true
    }

    /// Uses the Clark-Wright savings algorithm to attempt to generate a
    /// feasible solution.
    pub fn cw_savings(&mut self) -> bool {
        // For each pair of customers, compute the savings
        let mut savings = vec![vec![0.0; self.vrp.n_customers + 1]; self.vrp.n_customers + 1];
        for i in 1..(self.vrp.n_customers + 1) {
            for j in (i + 1)..(self.vrp.n_customers + 1) {
                let dist =
                    self.vrp.dist_mtx[0][i] + self.vrp.dist_mtx[j][0] - self.vrp.dist_mtx[i][j];
                savings[i][j] = dist;
                savings[j][i] = dist;
            }
        }

        // Sort savings by descending order
        let mut savings_heap = BinaryHeap::new();
        for i in 1..(self.vrp.n_customers + 1) {
            for j in (i + 1)..(self.vrp.n_customers + 1) {
                savings_heap.push((F64::from(savings[i][j]), i, j));
            }
        }

        // Process savings list until empty:
        // - If both customers are not in a route, create a new route
        // - If one customer is

        // If we have unassigned customers:
        // - If vehicles are available, greedily assign using NN
        // - Otherwise, find in existing routes the best place to insert each unassigned
        //   customer

        todo!()
    }
}

pub struct ProxIndividual {
    pub prox: f64,
    // TODO: maybe replace this with a slotmap key
    pub ind: Arc<Individual>,
}
