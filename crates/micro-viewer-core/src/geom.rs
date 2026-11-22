//! 3-D geometry: affine maps of (x, y, z), the camera's plane in the world, and turns.
//!
//! Axes follow the image: x to the right, y down (rows), z into the screen (deeper
//! planes). A camera looks along its normal = right x down.

pub type V3 = [f64; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn mul(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 {
        mul(a, 1.0 / n)
    } else {
        a
    }
}

/// `v` turned by `degrees` about the unit axis `n` (Rodrigues). Seen along `n` with y
/// down, a positive angle turns clockwise (as [`crate::Affine2::rotation`]).
pub fn turn(v: V3, n: V3, degrees: f64) -> V3 {
    let (s, c) = degrees.to_radians().sin_cos();
    add(
        add(mul(v, c), mul(cross(n, v), s)),
        mul(n, dot(n, v) * (1.0 - c)),
    )
}

/// A 3-D affine map of (x, y, z): `[x', y', z'] = m * [x, y, z, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine3 {
    pub m: [[f64; 4]; 3],
}

impl Affine3 {
    pub const IDENTITY: Affine3 = Affine3 {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    pub fn translation(t: V3) -> Self {
        let mut a = Self::IDENTITY;
        for (r, row) in a.m.iter_mut().enumerate() {
            row[3] = t[r];
        }
        a
    }

    /// A turn by `degrees` about the unit axis `n` through `about`.
    pub fn rotation(n: V3, degrees: f64, about: V3) -> Self {
        let cols = [
            turn([1.0, 0.0, 0.0], n, degrees),
            turn([0.0, 1.0, 0.0], n, degrees),
            turn([0.0, 0.0, 1.0], n, degrees),
        ];
        let mut a = Self::IDENTITY;
        for r in 0..3 {
            for (c, col) in cols.iter().enumerate() {
                a.m[r][c] = col[r];
            }
        }
        // about stays: t = about - R about
        let ra = a.linear(about);
        for (r, row) in a.m.iter_mut().enumerate() {
            row[3] = about[r] - ra[r];
        }
        a
    }

    /// The plane part (x, y) of a 2-D map, z left alone.
    pub fn from_2d(t: &crate::Affine2) -> Self {
        let m = &t.m;
        Self {
            m: [
                [m[0][0], m[0][1], 0.0, m[0][2]],
                [m[1][0], m[1][1], 0.0, m[1][2]],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// The x-y part (z taken as 0).
    pub fn to_2d(&self) -> crate::Affine2 {
        let m = &self.m;
        crate::Affine2 {
            m: [[m[0][0], m[0][1], m[0][3]], [m[1][0], m[1][1], m[1][3]]],
        }
    }

    pub fn apply(&self, p: V3) -> V3 {
        let l = self.linear(p);
        [
            l[0] + self.m[0][3],
            l[1] + self.m[1][3],
            l[2] + self.m[2][3],
        ]
    }

    pub fn linear(&self, p: V3) -> V3 {
        let m = &self.m;
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
        ]
    }

    /// `self` after `first`: p -> self(first(p)).
    pub fn after(&self, first: &Affine3) -> Self {
        let (a, b) = (&self.m, &first.m);
        let mut m = [[0.0; 4]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            }
            row[3] += a[r][3];
        }
        Self { m }
    }

    pub fn det(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        let m = &self.m;
        let c = |r0: usize, c0: usize, r1: usize, c1: usize| {
            m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
        };
        let inv = [
            [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
            [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
            [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
        ];
        let mut out = Self::IDENTITY;
        for (row, i) in out.m.iter_mut().zip(&inv) {
            row[..3].copy_from_slice(i);
            row[3] = -(i[0] * m[0][3] + i[1] * m[1][3] + i[2] * m[2][3]);
        }
        Some(out)
    }

    pub fn is_finite(&self) -> bool {
        self.m.iter().flatten().all(|v| v.is_finite())
    }

    /// Whether z comes out of z alone and x, y never see z (a map of planes to planes,
    /// e.g. any 2-D map).
    pub fn keeps_planes(&self, tol: f64) -> bool {
        let m = &self.m;
        m[2][0].abs() <= tol && m[2][1].abs() <= tol && m[0][2].abs() <= tol && m[1][2].abs() <= tol
    }
}

/// A plane in 3-D seen through the screen: the point at screen pixel (sx, sy) is
/// `origin + sx * ux + sy * uy` (pixel corners at integers, centres at + 0.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane3 {
    pub origin: V3,
    pub ux: V3,
    pub uy: V3,
}

impl Plane3 {
    pub fn at(&self, sx: f64, sy: f64) -> V3 {
        add(self.origin, add(mul(self.ux, sx), mul(self.uy, sy)))
    }

    /// The plane mapped through `a` (still a plane: affine maps keep planes).
    pub fn through(&self, a: &Affine3) -> Plane3 {
        Plane3 {
            origin: a.apply(self.origin),
            ux: a.linear(self.ux),
            uy: a.linear(self.uy),
        }
    }

    /// The x-y part as a 2-D map from the screen.
    pub fn to_2d(&self) -> crate::Affine2 {
        crate::Affine2 {
            m: [
                [self.ux[0], self.uy[0], self.origin[0]],
                [self.ux[1], self.uy[1], self.origin[1]],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine3_inverse_and_turns() {
        let a = Affine3 {
            m: [
                [0.5, 0.1, 0.2, 3.0],
                [-0.2, 2.0, 0.0, -7.0],
                [0.3, 0.0, 4.0, 1.5],
            ],
        };
        let b = a.inverse().unwrap();
        let p = b.apply(a.apply([11.0, -4.0, 2.5]));
        assert!(norm(sub(p, [11.0, -4.0, 2.5])) < 1e-12);
        // a turn about z by 90 degrees: x goes to y (clockwise with y down)
        let r = turn([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0);
        assert!(norm(sub(r, [0.0, 1.0, 0.0])) < 1e-12);
        let t = Affine3::rotation([0.0, 0.0, 1.0], 90.0, [10.0, 20.0, 5.0]);
        assert!(norm(sub(t.apply([11.0, 20.0, 5.0]), [10.0, 21.0, 5.0])) < 1e-12);
        assert!(t.keeps_planes(1e-12));
        let tilt = Affine3::rotation([1.0, 0.0, 0.0], 20.0, [0.0; 3]);
        assert!(!tilt.keeps_planes(1e-9));
        let two = crate::Affine2::rotation(30.0, 4.0, 2.0);
        let back = Affine3::from_2d(&two).to_2d();
        assert_eq!(back, two);
    }
}
