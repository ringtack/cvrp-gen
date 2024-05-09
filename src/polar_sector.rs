/// Represent a sector of a polar graph (start / end angle, in degrees scaled to [0, 65536) for modulo arithmetic).
pub struct PolarSector {
  pub start: i32,
  pub end: i32,
}

impl PolarSector {
  /// Constructs a polar sector from a single point.
  pub fn new(pt: i32) -> PolarSector {
    PolarSector { start: pt, end: pt }
  }

  /// Extends the sector to include a point.
  pub fn extend(&mut self, pt: i32) {
    if !self.encloses(pt) {
      if pos_mod(pt - self.end) <= pos_mod(self.start - pt) {
        self.end = pt;
      } else {
        self.start = pt;
      }
    }
  }

  /// Constructs a polar sector from a list of points.
  pub fn from_points(pts: &[i32]) -> PolarSector {
    let mut sector = PolarSector::new(pts[0]);
    for &pt in pts.iter().skip(1) {
      sector.extend(pt);
    }
    sector
  }

  /// Checks if a point is enclosed by this polar sector.
  pub fn encloses(&self, pt: i32) -> bool {
    pos_mod(pt - self.start) <= pos_mod(self.end - self.start)
  }

  /// Checks if this sector overlaps the other sector.
  pub fn overlaps(&self, other: &PolarSector) -> bool {
    pos_mod(other.start - self.start) <= pos_mod(self.end - self.start) || pos_mod(self.start - other.start) <= pos_mod(other.end - other.start)
  }
}


pub fn pos_mod(degrees: i32) -> i32 {
  (degrees % 65536 + 65536) % 65536
}
