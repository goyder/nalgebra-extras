extern crate nalgebra as na;
extern crate num_complex;

mod matrix_inverse;
mod matrix_norm;

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_inverse::NeumannInverse;
    use matrix_norm::MatrixNorm;

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
    #[test]
    fn test_neumann_inverse() {
        let epsilon = 0.00001;

        let a = na::Matrix3::new(0.9, -0.2, -0.3, 0.1, 1.0, -0.1, 0.3, 0.2, 1.1);

        let a_actual_inv = na::Matrix3::new(
            1.0,
            0.14285714,
            0.28571429,
            -0.125,
            0.96428571,
            0.05357143,
            -0.25,
            -0.21428571,
            0.82142857,
        );

        let a_inv = a.neumann_inverse(epsilon, 50).unwrap();
        assert!((a_inv - a_actual_inv).lp_norm(1) < epsilon);
    }
}
