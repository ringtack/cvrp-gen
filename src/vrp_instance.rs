use std::f64::consts::PI;
use std::fmt::Display;
use std::{fs, io::{self, BufRead}, time::Instant};

use kiddo::distance_metric::DistanceMetric;
use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use ordered_float::OrderedFloat;

use crate::params::Params;
use crate::polar_sector;

pub type F64 = OrderedFloat<f64>;

/// A VRP Instance.
#[derive(Debug, Clone)]
pub struct VRPInstance {
  /// Instance name.
  pub instance_name: String,
  /// Problem start time.
  pub start_time: Instant,

  /// Problem instance. Note that n_customers *excludes* the depot.
  pub n_customers: usize,
  pub n_vehicles: usize,
  pub vehicle_cap: usize,
  pub customers: Vec<Customer>,

  /// Nearest neighbors for each customer
  pub customer_nns: Vec<Vec<Customer>>,
  /// Distance matrix (squared) from one customer to another
  pub dist_mtx: Vec<Vec<f64>>,
  /// KDTree for nearest neighbor search
  pub kdtree: KdTree<f64, usize, 2, 32, u16>,

  /// Aggregate computations from customers
  pub max_demand: f64,
  pub total_demand: f64,

  /// Parameters used in the instance.
  pub params: Params,
}

impl VRPInstance {
  pub fn new(file: String, params: Params) -> VRPInstance {
    // Record algorithm start time
    let start_time = Instant::now();

    // Get basename from file path supplied
    let instance_name = file.split('/').last().unwrap().to_string();

    // Open the instance file for reading
    let f = fs::File::open(file).unwrap();
    // Parse first line, consisting of three integers
    let mut lines = io::BufReader::new(f).lines();
    let line = lines.next().unwrap().unwrap();
    let parts = line.split_whitespace().map(|x| x.parse::<usize>().unwrap()).collect::<Vec<_>>();
    // -1 to exclude depot
    let (n_customers, n_vehicles, vehicle_cap) = (parts[0]-1, parts[1], parts[2]);

    // Parse depot coordinates
    let line = lines.next().unwrap().unwrap();
    let parts = line.split_whitespace().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
    let (depot_x, depot_y) = (parts[0], parts[1]);

    // Represent 0 customer as depot
    let mut customers = Vec::with_capacity(n_customers+1);
    let mut id = 0;
    customers.push(Customer { id, x: depot_x, y: depot_y, demand: 0., polar_angle: 0 });
    id += 1;

    let mut max_demand: f64 = 0.;
    let mut total_demand = 0.;
    // For remaining lines, parse customers
    for line in lines {
      let line = line.unwrap();
      let mut parts = line.split_whitespace();
      // Read demand, x, and y from line
      let demand = parts.next().unwrap().parse::<f64>().unwrap();
      let x = parts.next().unwrap().parse::<f64>().unwrap();
      let y = parts.next().unwrap().parse::<f64>().unwrap();

      // Compute polar angle from depot
      let polar_angle = 32768. * (y - depot_y).atan2(x - depot_x) / PI;
      let polar_angle = polar_sector::pos_mod(polar_angle as i32);

      customers.push(Customer { id, x, y, demand, polar_angle });
      id += 1;

      max_demand = max_demand.max(demand);
      total_demand += demand;
    }

    // Compute distance matrix from one customer to another
    let mut dist_mtx = Vec::with_capacity(n_customers+1);
    for i in 0..(n_customers+1) {
      let mut row = Vec::with_capacity(n_customers+1);
      for j in 0..(n_customers+1) {
        let (x1, y1) = (customers[i].x, customers[i].y);
        let (x2, y2) = (customers[j].x, customers[j].y);
        let dist = SquaredEuclidean::dist(&[x1, y1], &[x2, y2]).sqrt();
        row.push(dist);
      }
      dist_mtx.push(row);
    }

    // Build KDTree from customer coords
    let c_coords = customers.iter().cloned().enumerate().map(|(i, c)| ([c.x, c.y], i)).collect::<Vec<_>>();
    let mut kdtree: KdTree<f64, usize, 2, 32, u16> = KdTree::with_capacity(c_coords.len());
    for (coord, idx) in c_coords {
      kdtree.add(&coord, idx);
    }

    // Compute nearest neighbors for each customer
    let mut customer_nns = Vec::with_capacity(customers.len());
    for c in customers.iter() {
      let (x, y) = (c.x, c.y);
      let mut nn = kdtree.nearest_n::<SquaredEuclidean>(&[x, y], params.gamma).iter().map(|x| customers[x.item]).collect::<Vec<_>>();
      // Remove depot from NN list
      nn.retain(|cs| cs.id != 0 && c.id != cs.id);

      customer_nns.push(nn);

    }

    VRPInstance {
      instance_name,
      start_time,
      n_customers,
      n_vehicles,
      vehicle_cap,
      customers,
      customer_nns,
      dist_mtx,
      max_demand,
      total_demand,
      params,
      kdtree,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Customer {
  pub id: usize,
  pub x: f64,
  pub y: f64,
  pub demand: f64,
  // Polar angle of client around depot the depot, in truncated degrees. For degree computation, see
  // https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/.
  pub polar_angle: i32,
}

impl Display for Customer {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Customer({}, {}, {}, {}, {})", self.id, self.x, self.y, self.demand, self.polar_angle)
  }
}