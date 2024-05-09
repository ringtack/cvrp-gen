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

use crate::{hgsls::HGSLS, local_search::LocalSearch};

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
        .filter(None, log::LevelFilter::Info)
        .init();

    log::info!("Starting CVRP solver on instance: {}", args.file);
    // Create VRP instance from file
    let vrp = Arc::new(VRPInstance::new(args.file, DEFAULT_PARAMS.clone()));

    // Create individual from VRP
    let mut ind = Individual::new(vrp.clone());
    // Solve using Bellman split
    let res = ind.bellman_split();

    if res {
        log::info!("Bellman split successful");

        log::debug!("Initial objective: {}", ind.objective());
        log::info!("Vehicle routes: {:?}", ind.routes);

        // Run local search on individual
        let mut ls = HGSLS::new(ind.clone(), DEFAULT_PARAMS.excess_penalty);
        let accept_temp = 0.1;
        let mut learned = ls.run(accept_temp, Duration::from_millis(5_000));

        log::info!("Initial objective: {}", ind.objective());
        log::info!("Local search complete; objective: {}", learned.objective());
        // learned.save_solution_default();

        let mut objs = Vec::new();
        objs.push(learned.objective);
        for _ in 0..10 {
            let mut best = learned.clone();
            // Save split solution
            best.bellman_split();

            log::info!(
                "Running local search again on split solution (obj: {})",
                best.objective()
            );
            // Run local search on split solution
            let mut ls = HGSLS::new(best.clone(), DEFAULT_PARAMS.excess_penalty);
            let accept_temp = 0.1;
            learned = ls.run(accept_temp, Duration::from_millis(5_000));
            log::info!("Total route: {:?}", learned.total_route);
            log::info!("Local search complete; objective: {}", learned.objective);
            objs.push(learned.objective);
        }

        learned.save_solution_default();

        // Get solution string from routes
        let mut sol = String::new();
        for route in learned.routes.iter() {
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
        log::info!("Objectives: {:?}", objs);
        println!(
            "{}",
            format!(
                r#"{{"Instance": "{}", "Time": "{:.2?}", "Result": "{:.2}", "Solution": "{}"}}"#,
                vrp.instance_name,
                vrp.start_time.elapsed().as_secs_f64(),
                learned.objective,
                sol,
            )
        )
    }
}
