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
