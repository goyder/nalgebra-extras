extern crate nalgebra as na;
extern crate num_complex;

use na::{DimMin, RawStorageMut, SquareMatrix};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix,  RawStorage, Storage};
use num_complex::{Complex as C};

fn main() {
    let a = na::Matrix2::new(
        C::new(5., 0.),
        C::new(1., 0.),
        C::new(0., 0.),
        C::new(5., 0.),
    );
    println!("2-norm of A: {}", a.norm());
}

#[derive(Debug)]
enum InverseError {
    IterationsReachedWithoutConvergence,
    InvalidNorm,
}

trait NeumannInverse<R, S>
where
    R: Dim,
    DefaultAllocator: Allocator<f64, R, R>,
{
    fn neumann_inverse(
        &self,
        epsilon: f64,
        max_iter: usize,
    ) -> Result<OMatrix<f64, R, R>, InverseError>;

    fn matrix_norm_inf(&self) -> f64;
}

impl<R, S> NeumannInverse<R, S> for SquareMatrix<f64, R, S>
where
    R: Dim + DimMin<R, Output=R>,
    DefaultAllocator: Allocator<f64, R>,
    DefaultAllocator: Allocator<f64, R, R>,
    S: Storage<f64, R, R>
        + RawStorage<f64, R, R>
        + RawStorageMut<f64, R, R>
        + Clone,
{
    fn neumann_inverse(
        &self,
        epsilon: f64,
        max_iter: usize,
    ) -> Result<OMatrix<f64, R, R>, InverseError> {
        let mut i = self.clone();
        i.fill_with_identity();

        // Calculate a and assess if we can calculate the inverse this way
        let a = (i - self).into_owned();
        if a.matrix_norm_inf() > 1. {
            return Err(InverseError::InvalidNorm)
        }

        let mut a_k_sum = self.clone_owned();
        let mut convergence_reached = false;
        a_k_sum.fill_with_identity();
        'k_summing: for k in 1..max_iter {
            let a_k = a.pow(k as u32);
            a_k_sum += a_k.clone();
            let delta = a_k.lp_norm(1);
            if delta < epsilon {
                convergence_reached = true;
                break 'k_summing
            }
        }

        if convergence_reached {
            return Ok(a_k_sum.into_owned())
        } else {
            return Err(InverseError::IterationsReachedWithoutConvergence)
        }
    }

    fn matrix_norm_inf(&self) -> f64 {
        let mut max_column_sum: f64 = 0.;

        for (index, col) in self.column_iter().enumerate() {
            let column_sum = col.iter().map(|&x| x.abs()).sum();
            if index == 0 {
                max_column_sum = column_sum;
            } else if column_sum > max_column_sum {
                max_column_sum = column_sum;
            }
        }

        max_column_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inf_norm_square() {
        let a = na::Matrix3::new(1., 2., 3., 4., 5., 6., 7., 8., 9.);

        assert_eq!(a.matrix_norm_inf(), 18.);
    }

    #[test]
    fn test_inf_norm_square_with_negative_components() {
        let a = na::Matrix3::new(1., 2., 3., 4., 5., 6., 7., 8., -9.);

        assert_eq!(a.matrix_norm_inf(), 18.);
    }

    #[test]
    fn test_neumann_inverse() {
        let epsilon = 0.00001;

        let a = na::Matrix3::new(
            0.9, -0.2, -0.3,
            0.1, 1.0, -0.1,
            0.3, 0.2, 1.1);

        let a_actual_inv = na::Matrix3::new(
            1.0, 0.14285714, 0.28571429,
            -0.125, 0.96428571, 0.05357143,
            -0.25, -0.21428571, 0.82142857);

        let a_inv = a.neumann_inverse(epsilon, 50).unwrap();
        assert!((a_inv - a_actual_inv).lp_norm(1) < epsilon);
    }
}
