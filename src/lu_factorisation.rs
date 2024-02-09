use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, OMatrix, RawStorageMut, SquareMatrix, StorageMut};

#[derive(Debug)]
pub enum LUFactorisationMethod {
    Doolittle,
    Crout,
}

#[derive(Debug)]
pub enum LUFactorisationError {
    GeneralError,
}

pub(crate) trait LUFactorisation<R>
where
    R: Dim,
    DefaultAllocator: Allocator<f64, R, R>,
{
    fn lu_factorisation(
        &self,
        factorisation_method: LUFactorisationMethod,
    ) -> Result<(OMatrix<f64, R, R>, OMatrix<f64, R, R>), LUFactorisationError>;
}

impl<R, S> LUFactorisation<R> for SquareMatrix<f64, R, S>
where
    R: Dim + DimMin<R, Output = R>,
    DefaultAllocator: Allocator<f64, R>,
    DefaultAllocator: Allocator<f64, R, R>,
    S: StorageMut<f64, R, R> + RawStorageMut<f64, R, R> + Clone,
{
    fn lu_factorisation(
        &self,
        factorisation_method: LUFactorisationMethod,
    ) -> Result<(OMatrix<f64, R, R>, OMatrix<f64, R, R>), LUFactorisationError> {
        let dimension = self.ncols();
        let mut l =
            OMatrix::<f64, R, R>::zeros_generic(R::from_usize(dimension), R::from_usize(dimension));
        let mut u =
            OMatrix::<f64, R, R>::zeros_generic(R::from_usize(dimension), R::from_usize(dimension));

        // Initialise the factorisation
        for i in 0..dimension {
            // Set l_ii and u_ii
            let mut liiuii_sum = 0.;
            if i > 0 {
                liiuii_sum = l
                    .view((i, 0), (1, i))
                    .iter()
                    .zip(u.view((0, i), (i, 1)).iter())
                    .map(|(r, c)| r * c)
                    .sum();
            };
            match factorisation_method {
                LUFactorisationMethod::Doolittle => {
                    u[(i, i)] = self[(i, i)] - liiuii_sum;
                    l[(i, i)] = 1.;
                }
                LUFactorisationMethod::Crout => {
                    u[(i, i)] = 1.;
                    l[(i, i)] = self[(i, i)] - liiuii_sum;
                }
            }

            // Solve for column i and row i
            for j in (i + 1)..dimension {
                let u_ij_sum: f64 = l
                    .view((i, 0), (1, i))
                    .iter()
                    .zip(u.view((0, j), (i, 1)).iter())
                    .map(|(r, c)| r * c)
                    .sum();
                u[(i, j)] = 1. / l[(i, i)] * (self[(i, j)] - u_ij_sum);

                let l_ji_sum: f64 = l
                    .view((j, 0), (1, j))
                    .iter()
                    .zip(u.view((0, i), (i, 1)).iter())
                    .map(|(r, c)| r * c)
                    .sum();

                l[(j, i)] = 1. / u[(i, i)] * (self[(j, i)] - l_ji_sum);
            }
        }

        Ok((l, u))
    }
}

#[cfg(test)]
mod lu_factorisation_tests {
    use super::*;

    #[test]
    fn test_doolittle_factorisation() {
        let epsilon = 1e-8;

        let a = na::Matrix3::new(0.9, -0.2, -0.3, 0.1, 1.0, -0.1, 0.3, 0.2, 1.1);

        let (l, u) = a
            .lu_factorisation(LUFactorisationMethod::Doolittle)
            .unwrap();

        assert!((a - l * u).lp_norm(1) < epsilon)
    }

    #[test]
    fn test_crout_factorisation() {
        let epsilon = 1e-8;

        let a = na::Matrix3::new(0.9, -0.2, -0.3, 0.1, 1.0, -0.1, 0.3, 0.2, 1.1);

        let (l, u) = a.lu_factorisation(LUFactorisationMethod::Crout).unwrap();

        assert!((a - l * u).lp_norm(1) < epsilon)
    }
}
