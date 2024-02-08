use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, OMatrix, RawStorageMut, SquareMatrix, StorageMut};

use crate::matrix_norm::MatrixNorm;

#[derive(Debug)]
pub enum InverseError {
    IterationsReachedWithoutConvergence,
    InvalidNorm,
}

pub(crate) trait NeumannInverse<R, S>
where
    R: Dim,
    DefaultAllocator: Allocator<f64, R, R>,
{
    fn neumann_inverse(
        &self,
        epsilon: f64,
        max_iter: usize,
    ) -> Result<OMatrix<f64, R, R>, InverseError>;
}

impl<R, S> NeumannInverse<R, S> for SquareMatrix<f64, R, S>
where
    R: Dim + DimMin<R, Output = R>,
    DefaultAllocator: Allocator<f64, R>,
    DefaultAllocator: Allocator<f64, R, R>,
    S: StorageMut<f64, R, R> + RawStorageMut<f64, R, R> + Clone,
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
            return Err(InverseError::InvalidNorm);
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
                break 'k_summing;
            }
        }

        if convergence_reached {
            return Ok(a_k_sum.into_owned());
        } else {
            return Err(InverseError::IterationsReachedWithoutConvergence);
        }
    }
}
