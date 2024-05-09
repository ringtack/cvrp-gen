use rand::rngs::SmallRng;
use lazy_static::lazy_static;

lazy_static! {
  // TODO: document values and configure
  pub static ref DEFAULT_PARAMS: Params = Params {
    mu: 16,
    lambda: 1000,
    gamma: 32,
    elite: 10,
    close: 10,
    excess_penalty: 10000.0,
    iter_ni: 20000,
    time_limit: 0,
    seed: 0,
  };
}

/// Genetic algorithm parameters.
/// Source: https://arxiv.org/pdf/2012.10384
#[derive(Clone, Debug)]
pub struct Params {
  /// Granularity in neighborhood search (for each customer, consider gamma_ NNs)
  pub gamma: usize,

  /// Minimum population size
  pub mu:  usize,
  /// Generation size
  pub lambda: usize,

  /// Number of "elite" individuals
  pub elite: usize,
  /// Number of "close" individuals to consider in diversity calculation
  pub close: usize,

  /// Penalty for exceeding capacity
  pub excess_penalty: f64,

  /// Iterations without improvement until termination (default: 20000)
  pub iter_ni: usize,
  /// Time limit until termination in ms (default 0)
  pub time_limit: usize,

  /// RNG seed
  pub seed: u64,
}