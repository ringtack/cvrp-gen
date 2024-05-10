use std::fmt::Display;

use lazy_static::lazy_static;

lazy_static! {
  pub static ref DEFAULT_PARAMS: Params = Params {
    // # NNs to consider
    gamma: 32,

    // Initial population size
    mu: 16,
    // Generation size
    lambda: 64,
    // Number of "elite" individuals
    elite: 4,
    // Number of "close" individuals to consider in diversity calculation
    close: 5,
    // Target feasible ratio
    xi: 0.2,
    // Runs before managing penalties
    penalty_runs: 100,
    // Penalty scaling
    penalty_inc: 1.2,
    penalty_dec: 0.8,
    // Initial penalty on excess capacity
    // TODO: scale by max(0.1, min(1_000, maxDist/maxDemand))
    excess_penalty: 1_000.0,

    // Number of no improvement iterations to run before stopping
    iter_ni: 20_000,
    // Max restarts before stopping
    max_restarts: 64,
    // Time limit in ms (TODO: change to 270s)
    time_limit: 10_000,
    // How often to print progress (in iterations)
    print_progress: 1000,

    // Default to system physical cores (to prevent interference from hyperthreading)
    n_threads: num_cpus::get_physical(),
  };
}

/// Genetic algorithm parameters.
/// Source: https://arxiv.org/pdf/2012.10384
#[derive(Copy, Clone, Debug)]
pub struct Params {
    /// Granularity in local search (for each customer, consider gamma NNs)
    pub gamma: usize,

    /// Minimum population size
    pub mu: usize,
    /// Generation size
    pub lambda: usize,
    /// Number of "elite" individuals
    pub elite: usize,
    /// Number of "close" individuals to consider in diversity calculation
    pub close: usize,
    /// Target ratio of feasible individuals in population
    pub xi: f64,
    /// How many runs before managing penalties
    pub penalty_runs: usize,
    /// How to scale penalties if over/under target ratio
    pub penalty_inc: f64,
    pub penalty_dec: f64,
    /// Penalty for exceeding capacity
    pub excess_penalty: f64,

    /// Iterations without improvement/restarts until termination (default:
    /// 20000)
    pub iter_ni: usize,
    pub max_restarts: usize,
    /// Time limit until termination in ms (default 0)
    pub time_limit: usize,

    /// How often to print progress. 0 is never.
    pub print_progress: usize,

    /// Number of threads available
    pub n_threads: usize,
}

impl Display for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "===== Genetic Algorithm Parameters =====")?;
        writeln!(f, "\t - Gamma (LS granularity): {}", self.gamma)?;
        writeln!(f, "\t - Mu (minimum population size): {}", self.mu)?;
        writeln!(f, "\t - Lambda (generation size): {}", self.lambda)?;
        writeln!(
            f,
            "\t - Elite (number of ELITE individuals): {}",
            self.elite
        )?;
        writeln!(
            f,
            "\t - Close (closest proximal individuals): {}",
            self.close
        )?;
        writeln!(f, "\t - xi (target feasible ratio): {}", self.xi)?;
        writeln!(
            f,
            "\t - penalty_runs (runs before scaling penalties): {}",
            self.penalty_runs
        )?;
        writeln!(
            f,
            "\t - penalty_inc (penalty increase scale): {}",
            self.penalty_inc
        )?;
        writeln!(
            f,
            "\t - penalty_dec (penalty decrease scale): {}",
            self.penalty_dec
        )?;
        writeln!(
            f,
            "\t - excess_penalty (initial excess cap penalty): {}",
            self.excess_penalty
        )?;
        writeln!(
            f,
            "\t - iter_ni (max iterations without improvement): {}",
            self.iter_ni
        )?;
        writeln!(f, "\t - time_limit (max time limit): {}", self.time_limit)?;
        writeln!(f, "\t - n_threads (n threads to used): {}", self.n_threads)
    }
}
