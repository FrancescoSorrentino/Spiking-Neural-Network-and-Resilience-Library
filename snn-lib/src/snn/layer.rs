use crate::snn::faults::{
    bit_flip, fault_iter, stuck_at_one, stuck_at_zero, ActualFault, Component, FaultType,
    OuterComponent,
};
use crate::snn::matrix_g::MatrixG;
use crate::snn::neuron::Neuron;

use std::fmt::{Display, Formatter};

#[derive(Clone)]
pub struct Layer<N: Neuron> {
    id: u32,
    neurons: Vec<N>,
    states: Vec<u8>,
    states_weights: Option<MatrixG<f64>>,
    weights: MatrixG<f64>,
}

impl<N: Neuron + Clone> Layer<N> {
    pub fn new(
        id: u32,
        neurons: u32,
        states_weights: Option<MatrixG<f64>>,
        weights: MatrixG<f64>,
        parameters: Option<&N::T>,
    ) -> Self {

        // Checks if weights dimensions (just rows) are consistent with given neuron number
        if weights.rows != neurons as usize {
            panic!("Invalid param, expected Weights matrix shape to be [{} , {}], but got [{}, {}] instead",
                   neurons,
                   weights.cols,
                   weights.rows,
                   weights.cols,
            );
        }
        // Checks if states weights dimensions (rows and cols) are consistent with given neuron number
        // and if the matrix's diagonal is 0
        if let Some(ref sw) = &states_weights {
            if sw.rows != neurons as usize || sw.cols != neurons as usize {
                panic!("Invalid param, expected States Weights matrix shape to be [{} , {}], but got [{}, {}] instead",
                       neurons,
                       neurons,
                       sw.rows,
                       sw.cols,
                );
            }
            for i in 0..sw.rows {
                for j in 0..sw.cols {
                    if i == j && sw[i][j] != 0.0 {
                        panic!("Invalid param, the diagonal of the States Weights matrix must be 0.0, but got {} instead", sw[i][j]);
                    }
                }
            }
        }

        let mut neurons_vec = Vec::<N>::with_capacity(neurons as usize);
        //Add matrices shape check??

        for i in 0..neurons {
            neurons_vec.push(N::new(i, parameters));
        }
        Layer {
            id,
            neurons: neurons_vec,
            states: vec![0; neurons as usize],
            states_weights,
            weights,
        }
    }

    //################# Utility Functions ##################

    pub fn get_id(&self) -> u32 {
        return self.id;
    }

    pub fn get_n_neurons(&self) -> usize {
        return self.neurons.len();
    }

    pub fn has_states_weights(&self) -> bool {
        self.states_weights.is_some()
    }

    pub fn set_neuron_parameters(&mut self, parameters: &N::T) {
        for n in self.neurons.iter_mut() {
            n.set_parameters(parameters);
        }
    }

    pub fn get_neuron_parameters(&self, idx: usize) -> N::T {
        return self.neurons[idx].get_parameters();
    }

    pub fn set_weights(&mut self, weights: MatrixG<f64>) {
        if weights.rows != self.neurons.len() {
            panic!("Invalid params, expected Weights vector shape to be [{} , {}], but got [{}, {}] instead",
                   self.neurons.len(),
                   weights.cols,
                   weights.rows,
                   weights.cols,
            )
        }
        self.weights = weights;
    }

    pub fn get_weights(&self) -> &MatrixG<f64> {
        return &self.weights;
    }

    pub fn set_states_weights(&mut self, weights: Option<MatrixG<f64>>) {
        if let Some(ref weights) = &weights {
            if weights.rows != self.neurons.len() || weights.cols != self.neurons.len() {
                panic!("Invalid params, expected Weights vector shape to be [{} , {}], but got [{}, {}] instead",
                       self.neurons.len(),
                       self.neurons.len(),
                       weights.rows,
                       weights.cols,
                )
            }
            for i in 0..weights.rows {
                for j in 0..weights.cols {
                    if i == j && weights[i][j] != 0.0 {
                        panic!("Invalid param, the diagonal of the States Weights matrix must be 0.0, but got {} instead", weights[i][j]);
                    }
                }
            }
        }
        self.states_weights = weights;
    }

    pub fn get_states_weights(&self) -> &Option<MatrixG<f64>> {
        return &self.states_weights;
    }

    //#########################################################

    pub fn forward(
        &mut self,
        inputs: &Vec<u8>,
        actual_fault: Option<&ActualFault<N::D>>,
        time: usize,
    ) -> Vec<u8> {
        let mut spikes = Vec::<u8>::with_capacity(self.neurons.len());

        match actual_fault {
            None => {
                for n in self.neurons.iter_mut() {
                    spikes.push(n.forward(
                        &inputs,
                        &self.states_weights,
                        &self.weights,
                        &self.states,
                        None,
                    ));
                }
            }
            Some(a_f) => {
                match a_f.component {
                    Component::Inside(_) => {
                        let fault = match a_f.fault_type {
                            FaultType::TransientBitFlip if a_f.time_tbf.unwrap() != time => None,
                            _ => actual_fault,
                        };
                        for n in self.neurons.iter_mut() {
                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                if a_f.neuron_id.0 == n.get_id() {
                                    fault
                                } else {
                                    None
                                },
                            ));
                        }
                    }
                    Component::Outside(ref c) => {
                        // true is for weights, false is for states_weights
                        let mut save = (true, 0.0);
                        let mut saved_weights = (Vec::<f64>::new(), Vec::<f64>::new());
                        match c {
                            OuterComponent::Weights => match a_f.fault_type {
                                FaultType::StuckAtZero if time == 0 => {
                                    stuck_at_zero(
                                        &mut self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::StuckAtOne if time == 0 => {
                                    stuck_at_one(
                                        &mut self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::TransientBitFlip => {
                                    if time == a_f.time_tbf.unwrap() {
                                        save = (
                                            true,
                                            self.weights[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize]
                                                .clone(),
                                        );
                                        bit_flip(
                                            &mut self.weights[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                }
                                _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
                                }
                            },
                            OuterComponent::Connections => {
                                //##### We suppose that both weights and internal weights are passed through the same buses ######//
                                match a_f.fault_type {
                                    FaultType::StuckAtZero if time == 0 => {
                                        fault_iter(
                                            &mut self.weights[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_zero,
                                        );
                                        if let Some(ref mut v) = self.states_weights {
                                            fault_iter(
                                                &mut v[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &stuck_at_zero,
                                            );
                                        }
                                    }
                                    FaultType::StuckAtOne if time == 0 => {
                                        fault_iter(
                                            &mut self.weights[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_one,
                                        );
                                        if let Some(ref mut v) = self.states_weights {
                                            fault_iter(
                                                &mut v[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &stuck_at_one,
                                            );
                                        }
                                    }
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            saved_weights.0 =
                                                self.weights[a_f.neuron_id.0 as usize].clone();
                                            fault_iter(
                                                &mut self.weights[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &bit_flip,
                                            );
                                            if let Some(ref mut v) = self.states_weights {
                                                saved_weights.1 =
                                                    v[a_f.neuron_id.0 as usize].clone();
                                                fault_iter(
                                                    &mut v[a_f.neuron_id.0 as usize],
                                                    a_f,
                                                    &bit_flip,
                                                );
                                            }
                                        }
                                    }
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
                                    }
                                }
                            }
                            OuterComponent::InnerWeights => match self.states_weights {
                                Some(ref mut sw) => match a_f.fault_type {
                                    FaultType::StuckAtZero if time == 0 => {
                                        stuck_at_zero(
                                            &mut sw[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::StuckAtOne if time == 0 => {
                                        stuck_at_one(
                                            &mut sw[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            save = (
                                                false,
                                                sw[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize]
                                                    .clone(),
                                            );
                                            bit_flip(
                                                &mut sw[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize],
                                                a_f.offset,
                                            );
                                        }
                                    }
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
                                    }
                                },
                                None => {}
                            },
                        }
                        for n in self.neurons.iter_mut() {
                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                None,
                            ));
                        }
                        match a_f.fault_type {
                            FaultType::TransientBitFlip if a_f.time_tbf.unwrap() == time => {
                                if a_f.bus.is_none() {
                                    if save.0 {
                                        self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize] = save.1
                                    } else {
                                        if let Some(ref mut v) = self.states_weights {
                                            v[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize] = save.1
                                        }
                                    }
                                } else {
                                    self.weights[a_f.neuron_id.0 as usize] = saved_weights.0;
                                    if let Some(ref mut v) = self.states_weights {
                                        v[a_f.neuron_id.0 as usize] = saved_weights.1;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        self.states = spikes.clone();

        spikes
    }
}

impl<N: Neuron> Display for Layer<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "| {:<16} | {:<16} | {:<16} | {:<16} |\n",
            self.id,
            self.neurons.len(),
            self.weights.rows * self.weights.cols,
            match &self.states_weights {
                None => 0,
                Some(w) => {
                    w.rows * w.rows - w.rows
                }
            }
        )
    }
}

