pub mod params;
pub mod vrp_instance;
pub mod genetic;
pub mod population;
pub mod individual;
pub mod local_search;
pub mod polar_sector;
pub mod ls_operators;

use std::{sync::Arc, time::Duration};

use clap::Parser;
use individual::*;
use params::DEFAULT_PARAMS;
use vrp_instance::VRPInstance;

use crate::local_search::LocalSearch;


#[derive(Parser, Debug, Clone)]
#[command(version, about = "CVRP Solver using hybrid genetic search.")]
pub struct Args {
  /// Path to the input file
  #[arg(short, long)]
  pub file: String,
}

fn main() {
  let args = Args::parse();

  log::set_max_level(log::LevelFilter::Trace);
  env_logger::builder()
    .filter(None, log::LevelFilter::Debug)
    .init();

  log::info!("Starting CVRP solver on instance: {}", args.file);
  // Create VRP instance from file
  let vrp = Arc::new(VRPInstance::new(args.file, DEFAULT_PARAMS.clone()));

  // Create individual from VRP
  let mut ind = Individual::new(vrp.clone());
  // Solve using Bellman split
  let res = ind.bellman_split();
  ind.objective();

  if res {
    log::info!("Bellman split successful");

    log::debug!("Initial objective: {}", ind.objective());
    log::info!("Vehicle routes: {:?}", ind.routes);

    // Run local search on individual
    let mut ls = LocalSearch::new(ind.clone());
    let accept_temp = 0.1;
    let mut learned = ls.run(accept_temp, Duration::from_millis(5_000), 5, 10_000.);

    log::info!("Initial objective: {}", ind.objective());
    log::info!("Local search complete; objective: {}", learned.objective());
    // learned.save_solution_default();

    let mut best = learned.clone();
    // Save split solution
    best.bellman_split();
    best.objective();

    log::info!("Running local search again on split solution");
    // Run local search on split solution
    let mut ls = LocalSearch::new(best.clone());
    let accept_temp = 0.1;
    let mut learned = ls.run(accept_temp, Duration::from_millis(5_000), 5, 10_000.);
    log::info!("Local search complete; objective: {}", learned.objective());
    learned.save_solution_default();

    log::info!("Total runtime: {:.2?}", vrp.start_time.elapsed());
  } else {
    log::error!("Bellman split failed");
  }
}
