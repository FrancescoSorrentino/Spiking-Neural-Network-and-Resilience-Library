use rand::distributions::uniform::SampleUniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct MatrixG<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<T>>,
}

impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> MatrixG<T> {
    // Return a matrix of all zeroes (or the default value of T)
    pub fn zeroes(rows: usize, cols: usize) -> MatrixG<T> {
        MatrixG {
            rows,
            cols,
            data: vec![vec![T::default(); cols]; rows],
        }
    }

    // Return a random generated matrix
    pub fn random(
        rows: usize,
        cols: usize,
        diag: bool,
        seed: Option<u64>,
        mut a: T,
        mut b: T,
    ) -> MatrixG<T> {
        if a == b {
            panic!("Cannot generate a random matrix with limit a ({}) equals to limit b ({})", a, b);
        }
        else if a > b {
            let tmp = a;
            a = b;
            b = tmp;
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut res = MatrixG::zeroes(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if diag && i == j {
                    res.data[i][j] = T::default();
                } else {
                    res.data[i][j] = rng.gen_range(a..=b);
                }
            }
        }
        res
    }

    // Create a Matrix starting from a Vector of Vector
    pub fn from(data: Vec<Vec<T>>) -> MatrixG<T> {
        let len = data[0].len();
        for t in &data {
            if t.len() != len {
                panic!(
                    "Invalid MatrixG, expected MatrixG to be shape= [{}], but got shape= [{}] instead",
                    len,
                    t.len()
                )
            }
        }
        MatrixG {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }
}

impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> Debug for MatrixG<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "MatrixG<{}> {{\n{}\n}}",
            std::any::type_name::<T>(),
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<String>>()
                        .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> Display for MatrixG<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "MatrixG<{}> {{[\n{}] shape= ({}, {})}}",
            std::any::type_name::<T>(),
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<String>>()
                        .join(" "))
                .collect::<Vec<String>>()
                .join("\n"),
            self.rows,
            self.cols
        )
    }
}

// Allow MatrixG to be accessed with matrix[i] without explicitly referencing matrix.data
impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> Index<usize> for MatrixG<T> {
    type Output = Vec<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> IndexMut<usize>
    for MatrixG<T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
