use crate::snn::faults::ActualFault;
use std::fmt::Debug;
use crate::snn::matrix_g::MatrixG;

pub trait Neuron {
    type T: NeuronParameters;
    type D: SpecificComponent + Clone + Debug + Sync;
    fn new(id: u32, parameters: Option<&Self::T>) -> Self;
    fn set_parameters(&mut self, parameters: &Self::T) -> ();
    fn get_parameters(&self) -> Self::T;
    fn get_id(&self) -> u32;
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<MatrixG<f64>>,
        weights: &MatrixG<f64>,
        states: &Vec<u8>,
        actual_fault: Option<&ActualFault<Self::D>>,
    ) -> u8;
}

pub trait NeuronParameters {}

pub trait SpecificComponent {}
