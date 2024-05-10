use std::{
    sync::{
        atomic::{AtomicBool, Ordering::SeqCst},
        Arc,
    },
    time::{Duration, Instant},
};

use crossbeam::channel::bounded;
use rand::random;

use crate::{
    hgsls::HGSLS,
    local_search::{LocalSearch, LS_LIMIT_MS},
    params::Params,
    population::Population,
    vrp_instance::VRPInstance,
    Individual,
};

// Max backlog of individuals to consider
const BACKLOG: usize = 1 << 10;
// What fraction of total time limit to allocate for each local search.
const LS_LIMIT_FRAC: f64 = 0.05;

/// Primary driving structure of genetic search.
pub struct GeneticSearch {
    /// VRP Instance to solve.
    vrp: Arc<VRPInstance>,
    /// Params (for easier access)
    params: Params,

    /// Population management.
    population: Population,

    /// Record objectives of best individuals found when restarts happen
    best_objectives: Vec<f64>,

    /// Record number of restarts w/o improvement
    n_restarts_ni: usize,
}

impl GeneticSearch {
    /// Create a new GeneticSearch instance.
    pub fn new(vrp: VRPInstance) -> GeneticSearch {
        let vrp = Arc::new(vrp);
        let ls_time_limit = Duration::from_millis(
            LS_LIMIT_MS.max((vrp.params.time_limit as f64 * LS_LIMIT_FRAC) as u64),
        );
        let pop = Population::new(vrp.clone(), ls_time_limit);
        GeneticSearch {
            params: vrp.params.clone(),
            vrp,
            population: pop,
            best_objectives: Vec::new(),
            n_restarts_ni: 0,
        }
    }

    /// Run the genetic search algorithm with the provided parameters, returning
    /// the best result seen.
    pub fn run(&mut self) -> Individual {
        log::info!("===== RUNNING GENETIC ALGORITHM =====");
        log::info!("{}", self.params);

        // Generate initial population
        self.population.generate_initial_population();

        let done = Arc::new(AtomicBool::new(false));
        // Create channel to send offspring to worker threads
        let (off_tx, off_rx) = bounded::<(Individual, f64)>(BACKLOG);
        // Create channel to receive offspring from worker threads
        let (res_tx, res_rx) = bounded::<Individual>(BACKLOG);

        // Initialize a bunch of offspring to evaluate
        let max_pop = self.params.mu + self.params.lambda;
        let n_offspring = (BACKLOG / 8).min(max_pop / 4);
        let offspring = self.generate_offspring(n_offspring);
        let excess_penalty = self.population.excess_penalty;
        // Send offspring to worker threads
        for off in offspring {
            off_tx.send((off, excess_penalty)).unwrap();
        }

        // Spawn worker threads to evaluate offspring; keep one thread for main genetic
        // algorithm
        let n_threads = self.params.n_threads - 1;
        // Choose LS time limit as max(LS_LIMIT_MS, 2.5% of total time limit)
        let ls_time_limit = Duration::from_millis(
            LS_LIMIT_MS.max((self.params.time_limit as f64 * LS_LIMIT_FRAC) as u64),
        );
        log::debug!("LS time limit: {:?}", ls_time_limit);
        log::debug!("n threads: {}", n_threads);

        let mut thrs = Vec::new();
        for _ in 0..n_threads {
            let off_rx = off_rx.clone();
            let res_tx = res_tx.clone();
            let ls_time_limit = ls_time_limit.clone();
            let done = done.clone();
            let thr_handle = std::thread::spawn(move || {
                // For each offspring to evaluate, run local search, then send back to
                // population
                // TODO: record average time waiting for recv, send, and backlog sizes
                while !done.load(SeqCst) {
                    if let Ok((ind, excess_penalty)) = off_rx.try_recv() {
                        let mut ls = HGSLS::new(ind);
                        let learned = ls.run(ls_time_limit, excess_penalty, 0.1);
                        // Don't block on sending result back, in case main thread backlogged
                        if let Err(_) = res_tx.try_send(learned) {
                            log::warn!("Failed to send offspring result back; channel full");
                        }
                    }
                }
            });
            thrs.push(thr_handle);
        }

        let start = Instant::now();
        let has_limit = self.params.time_limit > 0;
        let limit = Duration::from_millis(self.params.time_limit as u64);
        // Track iterations
        let mut i = 0;
        let mut i_ni = 0;
        loop {
            // If limit is set, check if we have exceeded it
            if has_limit && start.elapsed() > limit {
                log::info!("Time limit reached; terminating.");
                break;
            }
            // If no improvement for iter_ni iterations, check if within time limit
            if i_ni >= self.params.iter_ni {
                if has_limit && start.elapsed() < limit {
                    // Restart population search
                    let best_restart = self.population.restart();
                    self.best_objectives.push(best_restart.objective);
                    if best_restart.objective < self.population.best.objective {
                        self.n_restarts_ni = 0;
                    } else {
                        self.n_restarts_ni += 1;
                        if self.n_restarts_ni >= self.params.max_restarts {
                            log::info!("Max restarts reached. Terminating.");
                            break;
                        }
                    }
                } else {
                    log::info!("No improvement for {} iterations. Terminating.", i_ni);
                    break;
                }
            }

            log::trace!(
                "Backlog: {} (sending more if < {})",
                off_tx.len(),
                BACKLOG - n_offspring
            );
            // If worker channel has space, generate and send offspring
            if off_tx.len() < BACKLOG - n_offspring {
                let offspring = self.generate_offspring(n_offspring);
                let excess_penalty = self.population.excess_penalty;
                for off in offspring {
                    off_tx.send((off, excess_penalty)).unwrap();
                }
            }

            // Wait for at least one offspring to be received
            let mut num_rcvd = 0;
            let mut ind = res_rx.recv().unwrap();
            let best = self.population.add_individual(ind.clone(), true);
            i_ni = if best { 0 } else { i_ni + 1 };
            num_rcvd += 1;
            // If infeasible, randomly regen
            let feasible = ind.is_feasible();
            if !feasible && random::<bool>() {
                ind.bellman_split(1.5);
                let mut ls = HGSLS::new(ind);
                let learned = ls.run(ls_time_limit, self.population.excess_penalty, 0.1);
                self.population.add_individual(learned, false);
            }

            // Receive any remaining
            while let Ok(ind) = res_rx.try_recv() {
                let mut ind = ind;
                // Add individual to population
                let best = self.population.add_individual(ind.clone(), true);
                // Update no improvement counter if best
                i_ni = if best { 0 } else { i_ni + 1 };
                num_rcvd += 1;
                // If infeasible, randomly regen
                let feasible = ind.is_feasible();
                if !feasible && random::<bool>() {
                    ind.bellman_split(1.5);
                    let mut ls = HGSLS::new(ind);
                    let learned = ls.run(ls_time_limit, self.population.excess_penalty, 0.1);
                    self.population.add_individual(learned, false);
                }
            }
            log::trace!("Received {} offspring from workers", num_rcvd);

            // Check if need to rescale penalties
            if self.population.need_penalty_management() {
                log::debug!(
                    "Managing penalties (before): {:?}",
                    self.population.excess_penalty
                );
                self.population.manage_penalty();
                log::debug!(
                    "Managing penalties (after): {:?}",
                    self.population.excess_penalty
                );
            }

            // Check if need to do survivor selection
            if self.population.need_survivor_selection() {
                log::debug!(
                    "Selecting survivors (f: {}, inf: {})",
                    self.population.feasible.len(),
                    self.population.infeasible.len()
                );
                self.population.select_survivors();
                log::debug!(
                    "Selected (f: {}, inf: {})",
                    self.population.feasible.len(),
                    self.population.infeasible.len()
                );
            }

            i += 1;
            // Log every params.print_progress iterations
            if self.params.print_progress > 0 && i % self.params.print_progress == 0 {
                log::info!("====== GENETIC ALGORITHM STATE (iter {i}, ni {i_ni}) ======");
                log::info!("\t- Num restarts: {}", self.best_objectives.len());
                log::info!("\t- Best objectives: {:?}", self.best_objectives);
                log::info!("{}", self.population);
            }
        }

        // Signal worker threads to stop
        done.store(true, SeqCst);
        // Wait for worker threads to finish
        for thr in thrs {
            thr.join().unwrap();
        }

        // Return best individual found
        self.population.best.clone()
    }

    /// Generates n offspring.
    fn generate_offspring(&mut self, n: usize) -> Vec<Individual> {
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            let update = i == 0;
            let (p1, p2) = self.population.select_parents(update);
            let offspring = crossover(&p1, &p2);
            res.push(offspring);
        }
        res
    }
}

/// Crossover two individuals to produce new individual.
pub fn crossover(p1: &Individual, p2: &Individual) -> Individual {
    // Randomly pick start/end indices in p1 for crossover
    let len = p1.total_route.len();
    let mut start = rand::random::<usize>() % len;
    let end = rand::random::<usize>() % len;

    // Mark which customers have already been seen
    let mut seen = vec![false; p1.vrp.n_customers + 1];
    let mut total_route = vec![0; len];
    // Copy from [start, end)
    while {
        let c = p1.total_route[start];
        total_route[start] = c;
        seen[c] = true;

        start = (start + 1) % len;
        start != end
    } {}

    // Fill remaining customers from p2
    for i in 0..len {
        // Start from the end of the copied segment in p2
        let i = (end + i) % len;
        let c = p2.total_route[i];
        // Only consider unseen customers
        if !seen[c] {
            total_route[start] = c;
            seen[c] = true;
            // Update position in total_route
            start = (start + 1) % len;
        }
    }

    // Use average of excess capacities of parents
    let excess_penalty = (p1.excess_penalty + p2.excess_penalty) / 2.0;

    // Create new individual from total_route, and run split to compute initial
    // routes
    let mut new_ind = Individual::from_total_route(p1.vrp.clone(), total_route.clone());
    new_ind.set_excess_penalty(excess_penalty);
    let mut cap = 1.1;
    // Try bellman split for increasing capacities until success
    while !new_ind.bellman_split(cap) {
        cap *= 1.1;
    }
    log::trace!("Crossover: start={}, end={}", start, end);
    log::trace!("p1 route: {:?}", p1.total_route);
    log::trace!("p2 route: {:?}", p2.total_route);
    log::trace!(
        "After crossover: {:?} (penalty {})",
        total_route,
        excess_penalty
    );

    new_ind
}
