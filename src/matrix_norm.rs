use na::{Dim, Matrix, RawStorageMut, StorageMut};

pub(crate) trait MatrixNorm {
    fn matrix_norm_inf(&self) -> f64;
}

impl<R, C, S> MatrixNorm for Matrix<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<f64, R, C> + RawStorageMut<f64, R, C>,
{
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
mod matrix_norm_tests {
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
    fn test_inf_norm_nonsquare_with_negative_components() {
        let a = na::Matrix3x4::new(1., 2., 3., 1., 4., 5., 6., 1., 7., 8., -9., 1.);

        assert_eq!(a.matrix_norm_inf(), 18.);
    }
}
