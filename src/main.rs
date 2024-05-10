pub mod genetic;
pub mod hgsls;
pub mod individual;
pub mod local_search;
pub mod params;
pub mod polar_sector;
pub mod population;
pub mod vrp_instance;

use std::{sync::Arc, time::Duration};

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
}

fn main() {
    let args = Args::parse();

    log::set_max_level(log::LevelFilter::Trace);
    env_logger::builder().filter(None, args.verbosity).init();

    log::info!("Starting CVRP solver on instance: {}", args.file);

    let mut params = DEFAULT_PARAMS.clone();
    // params.n_threads = 1;
    params.time_limit = 120_000;
    // Create VRP instance from file
    let vrp = VRPInstance::new(args.file, params.clone());

    // // Generate two individuals
    // let mut ind = Individual::new(Arc::new(vrp.clone()));
    // let mut ind2 = Individual::new(Arc::new(vrp.clone()));
    // ind.bellman_split(1.5);
    // ind2.bellman_split(1.5);

    // // Attempt crossover
    // let mut ind = crossover(&ind, &ind2);
    // log::info!("Crossover complete; objective: {}", ind.objective());

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

    // Create genetic search instance
    let mut gen = GeneticSearch::new(vrp.clone());

    // Run genetic and get best result
    let mut best = gen.run();

    best.save_solution_default();

    // Get solution string from routes
    let mut sol = String::new();
    for route in best.routes.iter() {
        sol.push_str(&format!(
            "0 {}",
            route
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ));
        sol.push_str(" 0 ");
    }
    log::info!("Total runtime: {:.3?}", vrp.start_time.elapsed());
    println!(
        "{}",
        format!(
            r#"{{"Instance": "{}", "Time": "{:.2?}", "Result": "{:.2}", "Solution": "{}"}}"#,
            vrp.instance_name,
            vrp.start_time.elapsed().as_secs_f64(),
            best.objective,
            sol,
        )
    )
}

// // Create individual from VRP
// let mut ind = Individual::new(vrp.clone());
// // Solve using Bellman split
// let res = ind.bellman_split(1.5);

// if res {
//     log::info!("Bellman split successful");

//     log::debug!("Initial objective: {}", ind.objective());
//     log::info!("Vehicle routes: {:?}", ind.routes);

//     // Run local search on individual
//     let mut ls = HGSLS::new(ind.clone(), DEFAULT_PARAMS.excess_penalty);
//     let accept_temp = 0.1;
//     let mut learned = ls.run(accept_temp, Duration::from_millis(5_000));

//     log::info!("Initial objective: {}", ind.objective());
//     log::info!("Local search complete; objective: {}", learned.objective());
//     // learned.save_solution_default();

//     let mut objs = Vec::new();
//     objs.push(learned.objective);
//     for _ in 0..10 {
//         let mut best = learned.clone();
//         // Save split solution
//         best.bellman_split(1.5);

//         log::info!(
//             "Running local search again on split solution (obj: {})",
//             best.objective()
//         );
//         // Run local search on split solution
//         let mut ls = HGSLS::new(best.clone(), DEFAULT_PARAMS.excess_penalty);
//         let accept_temp = 0.1;
//         learned = ls.run(accept_temp, Duration::from_millis(5_000));
//         log::info!("Total route: {:?}", learned.total_route);
//         log::info!("Local search complete; objective: {}",
// learned.objective);         objs.push(learned.objective);
//     }
