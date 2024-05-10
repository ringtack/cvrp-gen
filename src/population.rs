use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt::Display,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crossbeam::channel;
use fxhash::FxHashMap;
use rand::Rng;
use slotmap::{new_key_type, SlotMap};

use crate::{hgsls::HGSLS, local_search::LocalSearch, vrp_instance::VRPInstance, Individual, INF};

// How much to tolerate differences from feasible:infeasible ratio
const RATIO_TOLERANCE: f64 = 0.025;
const PENALTY_LOWER_BOUND: f64 = 0.1;
const PENALTY_UPPER_BOUND: f64 = 10_000.;

new_key_type! {
  pub struct IndividualKey;
}

/// Population management.
pub struct Population {
    /// VRP Instance (and params).
    pub vrp: Arc<VRPInstance>,

    /// Best overall/per-restart solution
    pub best: Individual,
    pub best_restart: Individual,

    /// Feasible/infeasible population, sorted ascending by objective (so lower
    /// objective in front)
    pub feasible: Vec<IndividualKey>,
    pub infeasible: Vec<IndividualKey>,

    /// Proximal individuals (ind -> [(dist, ind)]), sorted ascending by
    /// distance (so lower distance in front)
    pub prox_individuals: FxHashMap<IndividualKey, Vec<(f64, IndividualKey)>>,

    /// TODO: add ind -> fitness??

    /// Recent individuals' feasibility
    pub recent_feasible: VecDeque<bool>,
    /// Tune for feasible/infeasible population management
    pub excess_penalty: f64,
    /// New individuals since last penalty tuning
    pub since_last_tune: usize,

    /// Search progress: times of best solution found, and their objective value
    pub search_progress: Vec<(Instant, f64)>,

    /// Local search time limit
    pub ls_time_limit: Duration,

    /// Internal slotmap to store individuals
    ind_slots: SlotMap<IndividualKey, Individual>,
}

impl Population {
    /// Create a new population with the provided VRP instance and local search
    /// time limit.
    pub fn new(vrp: Arc<VRPInstance>, ls_time_limit: Duration) -> Population {
        let excess_penalty = vrp.params.excess_penalty;
        let recent_feasible = VecDeque::from(vec![false; vrp.params.penalty_runs]);
        Population {
            vrp: vrp.clone(),
            best: Individual::new(vrp.clone()),
            best_restart: Individual::new(vrp),
            feasible: Vec::new(),
            infeasible: Vec::new(),
            prox_individuals: FxHashMap::default(),
            recent_feasible,
            excess_penalty,
            since_last_tune: 0,
            search_progress: Vec::new(),
            ls_time_limit,
            ind_slots: SlotMap::with_key(),
        }
    }

    /// Generates an initial population of size 4 * mu.
    pub fn generate_initial_population(&mut self) {
        let mu = self.vrp.params.mu;

        // Caps to scale w/ excess penalty if can't find feasible solution
        let excess_caps = [1., 1.1, 1.25, 1.5];

        // Create channel of 4 * mu individuals to generate
        let (idx_tx, idx_rx) = channel::bounded(4 * mu);
        // Insert i from [0..4 * mu) into channel
        for i in 0..4 * mu {
            idx_tx.send(i).unwrap();
        }

        // Create channel for workers to send individuals back
        let (res_tx, res_rx) = channel::bounded(4 * mu);
        // Spawn n_threads workers to generate initial solutions
        let n_threads = self.vrp.params.n_threads;
        for _ in 0..n_threads {
            let vrp = self.vrp.clone();
            let idx_rx = idx_rx.clone();
            let res_tx = res_tx.clone();
            let ls_time_limit = self.ls_time_limit;
            thread::spawn(move || {
                // Get individual index from channel
                while let Ok(i) = idx_rx.recv() {
                    log::trace!("Generating individual {}", i);
                    let mut ind = Individual::new(vrp.clone());
                    // Generate an initial solution, using Bellman split with varying levels
                    // of allowed penalties; if none work, default to parallel greedy NN.
                    // Every fourth individual, use greedy NN instead of Bellman split
                    if i % excess_caps.len() == 0 {
                        ind.greedy_nn();
                    } else {
                        let i_start = i % excess_caps.len() - 1;
                        // Try allowed excess caps from i_start.. end
                        let mut res = false;
                        for j in i_start..excess_caps.len() {
                            log::trace!(
                                "({i}) Trying Bellman split with excess cap: {}",
                                excess_caps[j]
                            );
                            res = ind.bellman_split(excess_caps[j]);
                            if res {
                                break;
                            }
                        }
                        // If none of them worked, default to greedy NN
                        if !res {
                            ind.greedy_nn();
                        }
                    }
                    // Run local search to optimize initial solution
                    let mut ls = HGSLS::new(ind);
                    let learned = ls.run(ls_time_limit, vrp.params.excess_penalty, 0.1);

                    log::trace!("Got learned individual w/ objective {}", learned.objective);
                    // Send learned individual back through channel
                    res_tx.send((learned, i)).unwrap();
                }
            });
        }

        // Collect individuals from workers
        for _ in 0..4 * mu {
            let (mut ind, i) = res_rx.recv().unwrap();
            self.add_individual(ind.clone(), true);
            // If infeasible, randomly re-run with higher penalty
            if !ind.is_feasible() && rand::random::<bool>() {
                log::debug!(
                    "Trying to repair infeasible individual from {i} (routes: {:?})",
                    ind.routes
                );
                // Check validity of individual
                assert!(ind.is_valid());
                ind.bellman_split(2.0);
                let mut ls = HGSLS::new(ind.clone());
                let learned = ls.run(self.ls_time_limit, self.excess_penalty * 10., 0.1); // Only add if feasible
                if learned.is_feasible() {
                    self.add_individual(learned, false);
                }
            }
        }
    }

    /// Restarts the population, clearing all individuals. Maintains overall
    /// best solution. Returns the restart's best solution
    pub fn restart(&mut self) -> Individual {
        log::debug!("===== RESTARTING POPULATION =====");

        let restart_best = self.best_restart.clone();

        // Reset best restart solution
        self.best_restart = Individual::new(self.vrp.clone());

        // Clear all individuals
        self.feasible.clear();
        self.infeasible.clear();

        // Clear all proximal calculations
        self.prox_individuals.clear();

        // Reset excess penalty and recent feasibility
        self.excess_penalty = self.vrp.params.excess_penalty;
        self.since_last_tune = 0;

        // Clear from slot map
        self.ind_slots.clear();

        // Generate new initial population
        self.generate_initial_population();

        restart_best
    }

    /// Selects two parents for offspring production.
    /// - If update is true, recompute fitnesses of both populations.
    pub fn select_parents(&mut self, update: bool) -> (Individual, Individual) {
        (self.select_parent(update), self.select_parent(false))
    }

    /// Randomly selects an individual from the population to be a parent,
    /// following the algorithm in Vidal 21.
    pub fn select_parent(&mut self, update: bool) -> Individual {
        // Recompute fitnesses of both populations to ensure accurate selection
        if update {
            self.update_pop_fitnesses(self.feasible.clone());
            self.update_pop_fitnesses(self.infeasible.clone());
        }

        // Get total population size
        let total_pop = self.feasible.len() + self.infeasible.len();
        // Randomly select two individuals between [0, total_pop)
        let i1 = rand::thread_rng().gen_range(0..total_pop);
        let i2 = rand::thread_rng().gen_range(0..total_pop);
        let ind1 = if i1 < self.feasible.len() {
            self.ind_slots[self.feasible[i1]].clone()
        } else {
            self.ind_slots[self.infeasible[i1 - self.feasible.len()]].clone()
        };
        let ind2 = if i2 < self.feasible.len() {
            self.ind_slots[self.feasible[i2]].clone()
        } else {
            self.ind_slots[self.infeasible[i2 - self.feasible.len()]].clone()
        };

        // Return the fittest of the two
        if ind1.fitness < ind2.fitness {
            ind1
        } else {
            ind2
        }
    }

    /// Adds an individual to the appropriate population. Returns true if
    /// solution is new (restart) best.
    pub fn add_individual(&mut self, ind: Individual, update_feasible: bool) -> bool {
        let feasible = ind.is_feasible();
        // Get slot for individual
        let ind_key = self.ind_slots.insert(ind);

        // Add to appropriate population, depending on feasibility
        self.insert_individual(ind_key);

        // Update proximal individuals for every individual in population
        let pop = if feasible {
            self.feasible.clone()
        } else {
            self.infeasible.clone()
        };
        for other_key in pop.iter() {
            // Insert default into proximal list if not already there
            self.prox_individuals
                .entry(ind_key)
                .or_insert_with(Vec::new);

            self.insert_proximal(ind_key, *other_key);
            self.insert_proximal(*other_key, ind_key);
        }

        // Update recent feasible list if specified
        if update_feasible {
            self.recent_feasible.push_back(feasible);
            self.recent_feasible.pop_front();
            self.since_last_tune += 1;
        }

        // Update best overall/since-restart solutions if feasible
        if feasible {
            let ind = &self.ind_slots[ind_key];
            if ind.objective < self.best_restart.objective {
                self.best_restart = ind.clone();
                if ind.objective < self.best.objective {
                    self.best = ind.clone();
                }
                // Update last time since best solution found
                self.search_progress.push((Instant::now(), ind.objective));
                return true;
            }
        }
        false
    }

    // Inserts an individual into its appropriate population.
    fn insert_individual(&mut self, ind_key: IndividualKey) {
        let ind = &self.ind_slots[ind_key];
        // Add to appropriate population, depending on feasibility
        let pop = if ind.is_feasible() {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        // Add individual to population: find index to insert at, by order of objective
        let idx = pop
            .binary_search_by(|x| {
                self.ind_slots[*x]
                    .objective
                    .partial_cmp(&ind.objective)
                    .unwrap()
            })
            .unwrap_or_else(|x| x);
        // Insert into population
        pop.insert(idx, ind_key);
    }

    /// Insert an individual other into indiviudal i's proximal list. Fails if
    /// other is already in i's proximal list.
    fn insert_proximal(&mut self, i_key: IndividualKey, other_key: IndividualKey) -> bool {
        let i = &self.ind_slots[i_key];
        let other = &self.ind_slots[other_key];
        // if self.prox_individuals.contains_key(&i_key) {
        //     assert!(false, "BUG: trying to insert duplicate individual!");
        //     return false;
        // }
        // Find best place to insert by distance
        let dist = broken_pairs_distance(i, other);
        let idx = self
            .prox_individuals
            .get(&i_key)
            .unwrap()
            .binary_search_by(|x| x.0.partial_cmp(&dist).unwrap())
            .unwrap_or_else(|x| x);
        // Insert into proximal list
        self.prox_individuals
            .get_mut(&i_key)
            .unwrap()
            .insert(idx, (dist, other_key));

        true
    }

    /// Check if need to change penalty based on recent feasibility.
    pub fn need_penalty_management(&mut self) -> bool {
        self.since_last_tune >= self.vrp.params.penalty_runs
    }

    /// Manage penalty to ensure feasibility ratio
    pub fn manage_penalty(&mut self) {
        // Scale current excess penalty based on fraction of recent feasible solutions
        let num_feasible = self.recent_feasible.iter().filter(|x| **x).count();
        let ratio = num_feasible as f64 / self.recent_feasible.len() as f64;

        let target_ratio = self.vrp.params.xi;
        // If ratio is too low (i.e. too few feasible), increase penalty for excess
        // capacity to discourage exploration
        if ratio < target_ratio - RATIO_TOLERANCE && self.excess_penalty >= PENALTY_LOWER_BOUND {
            self.excess_penalty =
                (self.excess_penalty * self.vrp.params.penalty_inc).max(PENALTY_LOWER_BOUND);
        }
        // If ratio is too high (i.e. too many feasible), decrease penalty for excess
        // capacity to encourage exploration
        if ratio > target_ratio + RATIO_TOLERANCE && self.excess_penalty <= PENALTY_UPPER_BOUND {
            self.excess_penalty =
                (self.excess_penalty * self.vrp.params.penalty_dec).min(PENALTY_UPPER_BOUND);
        }

        // Update objectives of infeasible individuals based on new penalty
        for i_key in self.infeasible.iter() {
            let ind = self.ind_slots.get_mut(*i_key).unwrap();
            ind.set_excess_penalty(self.excess_penalty);
            // Recompute objective
            ind.objective();
        }

        // Sort infeasible population by new objective
        self.infeasible.sort_by(|a, b| {
            let (a, b) = (&self.ind_slots[*a], &self.ind_slots[*b]);
            a.objective.partial_cmp(&b.objective).unwrap()
        });

        // Reset recent feasible count
        self.since_last_tune = 0;
    }

    /// Check if survivor selection of fittest is needed
    pub fn need_survivor_selection(&mut self) -> bool {
        let mu = self.vrp.params.mu;
        let lambda = self.vrp.params.lambda;
        let max_pop = mu + lambda;
        // If either subpopulation is too large, notify
        self.feasible.len() > max_pop || self.infeasible.len() > max_pop
    }

    /// Perform survivor selection on the population.
    pub fn select_survivors(&mut self) {
        let mu = self.vrp.params.mu;
        let lambda = self.vrp.params.lambda;
        let max_pop = mu + lambda;

        let mut pops = vec![self.feasible.clone(), self.infeasible.clone()];
        // If either population is too large, remove until mu left
        for pop in pops.iter_mut() {
            // If not yet at max size, skip
            if pop.len() < max_pop {
                continue;
            }

            // Remove until mu left
            while pop.len() > mu {
                // Re-compute fitnesses of each individual in population
                self.update_pop_fitnesses(pop.clone());

                // Find least fit individual (i.e. w/ highest fitness level)
                // TODO: see if I need to handle duplicates/clones? there really shouldn't
                // be any, but HGS-CVRP does it...
                let least_fit = *pop
                    .iter()
                    .max_by(|a, b| {
                        self.ind_slots[**a]
                            .fitness
                            .partial_cmp(&self.ind_slots[**b].fitness)
                            .unwrap()
                    })
                    .unwrap();

                pop.remove(pop.iter().position(|x| *x == least_fit).unwrap());
                // Remove from proximal individuals
                self.prox_individuals.remove(&least_fit);
                // From each proximal individual, remove if it has least_fit
                for (_, prox) in self.prox_individuals.iter_mut() {
                    prox.retain(|(_, i)| *i != least_fit);
                }
                // Remove from slotmap
                self.ind_slots.remove(least_fit);
            }
        }

        // Update pops
        self.feasible = pops[0].clone();
        self.infeasible = pops[1].clone();
    }

    /// Computes the holistic fitness of each individual in the population,
    /// weighted by both its average proximal distnace to its neighbors, plus
    /// its actual fitness rank in the population.
    fn update_pop_fitnesses(&mut self, pop: Vec<IndividualKey>) {
        // If just one element in population, it is most fit
        if pop.len() == 1 {
            self.ind_slots[pop[0]].fitness = 0.0;
            return;
        }

        // Otherwise, sort by average diversity (ascending) to neighbors; vector stores
        // (diversity value, fitness rank)
        let mut ranking = pop
            .iter()
            .enumerate()
            .map(|(fitness, i_key)| (self.get_diversity(*i_key, self.vrp.params.close), fitness))
            .collect::<Vec<_>>();
        // Sort by diversity, then by fitness
        ranking.sort_by(|a, b| {
            match a.0.partial_cmp(&b.0).unwrap() {
                Ordering::Less => Ordering::Less,
                Ordering::Equal => a.1.cmp(&b.1),
                Ordering::Greater => Ordering::Greater,
            }
        });

        // Index in ranking now determines diversity rank
        for (i, (_, fitness)) in ranking.iter().enumerate() {
            // Scale diversity rank from [0, 1]
            let diversity_rank = i as f64 / (pop.len() - 1) as f64;
            let fitness_rank = *fitness as f64 / (pop.len() - 1) as f64;

            self.ind_slots[pop[*fitness]].fitness = if pop.len() <= self.vrp.params.elite {
                // If all individuals are ELITE, just consider fitness rank
                fitness_rank
            } else {
                // Otherwise, scale by fitness rank and diversity rank: lower diversity rank ->
                // lower fitness rank (i.e. higher fitness)
                fitness_rank
                    + (1. - self.vrp.params.elite as f64 / pop.len() as f64) * diversity_rank
            };
        }
    }

    /// Gets the diversity of an individual (computed by avg distance to its
    /// proximal neighbors)
    fn get_diversity(&self, i_key: IndividualKey, n_close: usize) -> f64 {
        // Get individual's proximal neighbors
        let prox = self.prox_individuals.get(&i_key).unwrap();
        let size = prox.len().min(n_close);
        // Average distances of vrp.close proximal neighbors
        prox.iter().take(size).map(|x| x.0).sum::<f64>() / size as f64
    }

    /// Gets the best individual in the population.
    pub fn get_best(&self) -> Individual {
        self.best.clone()
    }

    /// Gets the best individual since restart in the population.
    pub fn get_best_restart(&self) -> Individual {
        self.best_restart.clone()
    }

    /// Gets the best feasible individual in the population, if it exists.
    pub fn get_best_feasible(&self) -> Option<Individual> {
        if self.feasible.is_empty() {
            None
        } else {
            Some(self.ind_slots[self.feasible[0]].clone())
        }
    }

    /// Gets the best infeasible individual in the population, if it exists.
    pub fn get_best_infeasible(&self) -> Option<Individual> {
        if self.infeasible.is_empty() {
            None
        } else {
            Some(self.ind_slots[self.infeasible[0]].clone())
        }
    }

    /// Gets the average objective of the feasible population.
    pub fn get_avg_objective_feasible(&self) -> f64 {
        if self.feasible.is_empty() {
            return INF;
        }
        self.feasible
            .iter()
            .map(|x| self.ind_slots[*x].objective)
            .sum::<f64>()
            / self.feasible.len() as f64
    }

    /// Gets the average objective of the infeasible population.
    pub fn get_avg_objective_infeasible(&self) -> f64 {
        if self.infeasible.is_empty() {
            return INF;
        }
        self.infeasible
            .iter()
            .map(|x| self.ind_slots[*x].objective)
            .sum::<f64>()
            / self.infeasible.len() as f64
    }

    /// Gets the average diversity of the feasible population.
    pub fn get_avg_diversity_feasible(&self) -> f64 {
        if self.feasible.is_empty() {
            return INF;
        }
        self.feasible
            .iter()
            .map(|x| self.get_diversity(*x, self.vrp.params.close))
            .sum::<f64>()
            / self.feasible.len() as f64
    }

    /// Gets the average diversity of the infeasible population.
    pub fn get_avg_diversity_infeasible(&self) -> f64 {
        if self.infeasible.is_empty() {
            return INF;
        }
        self.infeasible
            .iter()
            .map(|x| self.get_diversity(*x, self.vrp.params.close))
            .sum::<f64>()
            / self.infeasible.len() as f64
    }

    /// Gets the ratio of recent feasible solutions.
    pub fn get_recent_feasible_ratio(&self) -> f64 {
        self.recent_feasible.iter().filter(|x| **x).count() as f64
            / self.recent_feasible.len() as f64
    }

    /// Gets the current excess penalty.
    pub fn get_excess_penalty(&self) -> f64 {
        self.excess_penalty
    }
}

impl Display for Population {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let best_obj_feasible = self
            .get_best_feasible()
            .map_or("None".to_string(), |x| x.objective.to_string());

        let best_obj_infeasible = self
            .get_best_infeasible()
            .map_or("None".to_string(), |x| x.objective.to_string());

        writeln!(f, "===== POPULATION STATS:").unwrap();
        writeln!(f, "\t- Best Overall: {}", self.best.objective).unwrap();
        writeln!(f, "\t- Feasible:").unwrap();
        writeln!(f, "\t\t* Best: {}", best_obj_feasible).unwrap();
        writeln!(
            f,
            "\t\t* Avg. Objective: {}",
            self.get_avg_objective_feasible()
        )
        .unwrap();
        writeln!(
            f,
            "\t\t* Avg. Diversity: {}",
            self.get_avg_diversity_feasible()
        )
        .unwrap();
        writeln!(f, "\t- Infeasible:").unwrap();
        writeln!(f, "\t\t* Best: {}", best_obj_infeasible).unwrap();
        writeln!(
            f,
            "\t\t* Avg. Objective: {}",
            self.get_avg_objective_infeasible()
        )
        .unwrap();
        writeln!(
            f,
            "\t\t* Avg. Diversity: {}",
            self.get_avg_diversity_infeasible()
        )
        .unwrap();
        writeln!(
            f,
            "\t- Recent Feasible Ratio: {}",
            self.get_recent_feasible_ratio()
        )
        .unwrap();
        writeln!(f, "\t- Excess Penalty: {}", self.get_excess_penalty())
    }
}

/// Compute broken pairs distance between two individuals: sum of differences
/// between successors/predecessors of each customer in both routes.
pub fn broken_pairs_distance(i1: &Individual, i2: &Individual) -> f64 {
    let mut dist = 0;
    let n_customers = i1.vrp.n_customers;
    for c in 1..=n_customers {
        // If predecessor and successor of c is different, increment distance
        let (p1, s1) = (i1.pred[c], i1.succ[c]);
        let (p2, s2) = (i2.pred[c], i2.succ[c]);
        if p1 != p2 && s1 != s2 {
            dist += 1;
        }
        // If c's pred is depot, and neither pred nor succ of c is depot, increment
        // distance
        if p1 == 0 && p2 != 0 && s2 != 0 {
            dist += 1;
        }
    }
    // Compute distance as a fraction of total customers
    dist as f64 / n_customers as f64
}
