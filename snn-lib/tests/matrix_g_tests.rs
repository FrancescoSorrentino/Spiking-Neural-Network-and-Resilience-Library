#[cfg(test)]
mod matrix_g_tests {
    use snn_lib::snn::matrix_g::MatrixG;

    #[test]
    fn input_matrix_creation() {
        // Randomly creating an input matrix
        let input_matrix = MatrixG::random(2, 3, false, None, 0, 1);
        // Checks matrix dimensions
        assert_eq!(input_matrix.rows, 2);
        assert_eq!(input_matrix.cols, 3);
        assert_eq!(input_matrix.cols, input_matrix[0].len());
        assert_eq!(input_matrix.rows, input_matrix.data.len());

        // Checks that the generated values are inside the given range
        for i in 0..input_matrix.rows {
            for j in 0..input_matrix.cols {
                assert!(input_matrix[i][j] <= 1);
                assert!(input_matrix[i][j] >= 0);
            }
        }
    }

    #[test]
    fn diag_matrix_creation() {
        let matrix = MatrixG::random(2, 3, true, None, 1, 2);
        // Checks the diagonal, should be 0
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                if i == j {
                    assert_eq!(matrix[i][j], 0);
                }
            }
        }
    }

    #[test]
    fn weights_matrix_creation() {
        // Randomly creating an input matrix
        let weights_matrix = MatrixG::random(2, 3, false, None, 0.01, 0.99);

        // Checks that the generated values are inside the given range
        for i in 0..weights_matrix.rows {
            for j in 0..weights_matrix.cols {
                assert!(weights_matrix[i][j] <= 0.99);
                assert!(weights_matrix[i][j] >= 0.01);
            }
        }
    }

    #[test]
    fn wrong_matrix_creation_inverted_limits() {
        // Randomly creating an input matrix
        let weights_matrix = MatrixG::random(2, 3, false, None, 0.99, 0.01);
        for i in 0..weights_matrix.rows {
            for j in 0..weights_matrix.cols {
                assert!(weights_matrix[i][j] <= 0.99);
                assert!(weights_matrix[i][j] >= 0.01);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Cannot generate a random matrix with limit a (0.99) equals to limit b (0.99)")]
    fn wrong_matrix_creation_empty_range() {
        // Randomly creating an input matrix
        let _weights_matrix = MatrixG::random(2, 3, false, None, 0.99, 0.99);
    }

    #[test]
    fn matrix_from() {
        let v = vec![vec![0,1], vec![1,0]];
        let m = MatrixG::from(v);
        // Checks the generation of a MatrixG from a vec of vecs
        assert_eq!(m.cols, 2);
        assert_eq!(m.rows, 2);
        assert_eq!(m[0][0], 0);
        assert_eq!(m[0][1], 1);
        assert_eq!(m[1][0], 1);
        assert_eq!(m[1][1], 0);
    }

}