#![allow(unused_imports)]

pub mod genetic;
pub mod hgsls;
pub mod individual;
pub mod local_search;
pub mod params;
pub mod polar_sector;
pub mod population;
pub mod vrp_instance;

use std::sync::Arc;

use clap::Parser;
use individual::*;
use params::DEFAULT_PARAMS;
use vrp_instance::VRPInstance;

use crate::{
    genetic::{crossover, GeneticSearch},
    hgsls::HGSLS,
    local_search::LocalSearch,
};

#[derive(Parser, Debug, Clone)]
#[command(version, about = "CVRP Solver using hybrid genetic search.")]
pub struct Args {
    /// Path to the input file
    #[arg(short, long)]
    pub file: String,

    /// Logging level (trace, debug, info, warn, error)
    #[arg(short, long, default_value_t = log::LevelFilter::Info)]
    pub verbosity: log::LevelFilter,

    /// If solution file provided, try to optimize with LS
    #[arg(long)]
    pub solution: Option<String>,

    /// Number of threads. Default is number of physical cores.
    #[arg(long)]
    pub threads: Option<usize>,

    /// Time limit in seconds. Default is 270s.
    #[arg(long)]
    pub time_limit: Option<usize>,

    /// Max iterations without improvement. Default is 32_768.
    #[arg(long)]
    pub iter_ni: Option<usize>,

    /// Max restarts. Default is 32.
    #[arg(long)]
    pub restarts: Option<usize>,

    /// Minimum population size. Default is 24.
    #[arg(long)]
    pub mu: Option<usize>,

    /// Generation size. Default is 80.
    #[arg(long)]
    pub lambda: Option<usize>,
}

fn main() {
    let args = Args::parse();

    log::set_max_level(log::LevelFilter::Trace);
    env_logger::builder().filter(None, args.verbosity).init();

    log::info!("Starting CVRP solver on instance: {}", args.file);

    let mut params = DEFAULT_PARAMS.clone();
    // Change depending on args
    if let Some(threads) = args.threads {
        params.n_threads = threads;
    }
    if let Some(time_limit) = args.time_limit {
        params.time_limit = time_limit;
    }

    // Create VRP instance from file
    let vrp = VRPInstance::new(args.file, params.clone());

    if let Some(solution_file) = args.solution {
        let ind = Individual::load_solution(Arc::new(vrp), &solution_file);
        let mut ls = HGSLS::new(ind.clone());
        let mut learned = ls.run(
            std::time::Duration::from_millis(5_000),
            DEFAULT_PARAMS.excess_penalty,
            0.1,
        );
        learned.save_solution_default();
        return;
    }

    // Create genetic search instance
    let mut gen = GeneticSearch::new(vrp.clone());

    // Run genetic and get best result
    let mut best = gen.run();

    best.save_solution_default();

    // Get solution string from routes
    let mut sol = String::from("0 ");
    for route in best.routes.iter() {
        let mut route_str = route.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        route_str.insert(0, "0".to_string());
        route_str.push("0".to_string());
        sol = format!("{}{} ", sol, route_str.join(" "));
    }
    // Get rid of last space
    sol.pop();
    log::info!("Total runtime: {:.3?}", vrp.start_time.elapsed());
    println!(
        "{}",
        format!(
            r#"{{"Instance": "{}", "Time": {:.2?}, "Result": {:.2}, "Solution": "{}"}}"#,
            vrp.instance_name,
            vrp.start_time.elapsed().as_secs_f64(),
            best.objective,
            sol,
        )
    )
}

// // Use bellman split, and save solution
// let mut ind = Individual::new(Arc::new(vrp.clone()));
// // ind.bellman_split(1.);
// ind.greedy_nn();
// ind.save_solution("greedy.1");

// // LS on result
// let mut ls = HGSLS::new(ind.clone());
// let mut learned = ls.run(
//     std::time::Duration::from_millis(5_000),
//     DEFAULT_PARAMS.excess_penalty,
//     0.1,
// );
// learned.save_solution("hgsls_1.1");

// return;

// Generate two individuals
// let mut ind = Individual::new(Arc::new(vrp.clone()));
// let mut ind2 = Individual::new(Arc::new(vrp.clone()));
// ind.bellman_split(1.1);
// ind2.bellman_split(1.1);
// let mut ls1 = HGSLS::new(ind.clone());
// let mut ls2 = HGSLS::new(ind2.clone());
// let mut learned1 = ls1.run(
//     std::time::Duration::from_millis(5_000),
//     DEFAULT_PARAMS.excess_penalty,
//     0.1,
// );
// let mut learned2 = ls2.run(
//     std::time::Duration::from_millis(5_000),
//     DEFAULT_PARAMS.excess_penalty,
//     0.1,
// );
// learned1.save_solution("hgsls_1.1_1");
// learned2.save_solution("hgsls_1.1_2");

// // // Attempt crossover
// let mut ind = crossover(&learned1, &learned2);
// log::info!("Crossover complete; objective: {}", ind.objective());
// ind.save_solution("crossover");
// return;

// Run basic HGSLS on VRP instance
// let mut ind = Individual::new(Arc::new(vrp.clone()));
// ind.bellman_split(1.1);
// // ind.greedy_nn();
// let mut ls = HGSLS::new(ind.clone());
// let mut learned = ls.run(
//     Duration::from_millis(5_000),
//     DEFAULT_PARAMS.excess_penalty,
//     0.1,
// );
// let mut ind = learned.clone();
// let mut ls = HGSLS::new(ind.clone());
// learned = ls.run(
//     Duration::from_millis(5_000),
//     DEFAULT_PARAMS.excess_penalty * 10.,
//     0.1,
// );

// log::info!("Initial objective: {}", ind.objective());
// log::info!("Local search complete; objective: {}", learned.objective());

// log::info!("Total runtime: {:.3?}", vrp.start_time.elapsed());
