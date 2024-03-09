use crate::snn::faults::{add, apply_fault, compare, div, mul, ActualFault, Component, FaultType};
use crate::snn::matrix_g::MatrixG;
use crate::snn::neuron::NeuronParameters;
use crate::snn::neuron::{Neuron, SpecificComponent};

#[derive(Clone, Copy, Debug)]
pub enum ResetMode {
    Zero,
    RestingPotential,
    SubThreshold,
}

#[derive(Clone, Debug)]
pub struct LifNeuronParameters {
    pub v_rest: f64,
    pub v_th: f64,
    pub r_type: ResetMode,
    pub tau: f64,
}

impl NeuronParameters for LifNeuronParameters {}

#[derive(Clone, PartialEq, Debug)]
pub enum LifSpecificComponent {
    Adder,      // add
    Multiplier, // mul
    Divider,    // div
    Comparator, // compare
    Threshold,  // v_th
    Membrane,   // v_mem
    Rest,       // v_rest
}

impl SpecificComponent for LifSpecificComponent {}

#[derive(Clone)]
pub struct LifNeuron {
    id: u32,
    v_mem: f64,
    v_rest: f64,
    v_th: f64,
    r_type: ResetMode,
    t_s_last: f64, // time from last spike of the neuron -> t_s_last = t - t_last
    tau: f64,
    time_step: f64,
    broken: bool,
}

impl LifNeuron {
    fn scalar_product(
        inputs: &Vec<u8>,
        weights: &Vec<f64>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        ops: &Vec<bool>,
    ) -> f64 {
        let mut scalar = 0.0;

        for (idx, x) in inputs.iter().enumerate() {
            // Summation for each neuron
            scalar = add(
                scalar,
                mul(*x as f64, weights[idx], actual_fault, ops[1]),
                actual_fault,
                ops[0],
            ); // Multiply spike for input's weights
        }
        scalar
    }

    fn y(
        inputs: &Vec<u8>,
        states: &Vec<u8>,
        weights: &Vec<f64>,
        states_weights: Option<&Vec<f64>>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        ops: &Vec<bool>,
    ) -> f64 {
        // Scalar product between input and input's weights
        let out = LifNeuron::scalar_product(inputs, weights, actual_fault, ops);

        match states_weights {
            // if there is state weights array
            Some(states_weights) => {
                let sum1 = LifNeuron::scalar_product(states, states_weights, actual_fault, ops);
                add(out, sum1, actual_fault, ops[0])
            }
            // if there is no state weights array
            None => out,
        }
        // Out it's the linear combination between input array and array of associated weights
    }
}

impl Neuron for LifNeuron {
    type T = LifNeuronParameters;
    type D = LifSpecificComponent;
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the snn existence)
    fn new(id: u32, parameters: Option<&LifNeuronParameters>) -> Self {
        match parameters {
            Some(p) => {
                if p.tau <= 0.0 {
                    panic!("Invalid neuron params, tau must be strictly positive");
                }
                if p.v_th <= 0.0 {
                    panic!("Invalid neuron params, v_th must be strictly positive");
                }
                LifNeuron {
                    id,
                    v_mem: 0.0,
                    v_rest: p.v_rest,
                    v_th: p.v_th,
                    r_type: p.r_type,
                    t_s_last: 0.0,
                    tau: p.tau,
                    time_step: 1e-3,
                    broken: false,
                }
            }
            None => LifNeuron {
                id,
                v_mem: 0.0,
                v_rest: 0.0,
                v_th: 0.8,
                r_type: ResetMode::Zero,
                t_s_last: 0.0,
                tau: 5e-3, // 5ms it's a realistic biological neuron value for R=50MΩ  and C=100pF
                time_step: 1e-3, // We fix a time step for every call of the forward.
                broken: false,
            },
        }
    }

    fn set_parameters(&mut self, parameters: &LifNeuronParameters) {
        // Checks that values are not below or equal to 0
        if parameters.tau <= 0.0 {
            panic!("Invalid neuron params, tau must be strictly positive");
        }
        if parameters.v_th <= 0.0 {
            panic!("Invalid neuron params, v_th must be strictly positive");
        }
        self.v_rest = parameters.v_rest;
        self.v_th = parameters.v_th;
        self.r_type = parameters.r_type;
        self.tau = parameters.tau;
    }

    fn get_parameters(&self) -> Self::T {
        LifNeuronParameters {
            v_rest: self.v_rest,
            v_th: self.v_th,
            r_type: self.r_type,
            tau: self.tau,
        }
    }

    fn get_id(&self) -> u32 {
        self.id
    }

    // Implements the forward pass of the snn
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<MatrixG<f64>>,
        weights: &MatrixG<f64>,
        states: &Vec<u8>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
    ) -> u8 {

        self.t_s_last += self.time_step;

        // Stop neuron execution if no spike is present in input, to make the whole thing "event based",
        // actual calculations are made only when a spike is received.
        // We could also have moved this logic inside the layer and avoid calling the forward function at all
        // if no input spike is present, but to maintain neurons' data consistent during the execution, and
        // also to guarantee generalization (maybe another kind of neuron does something in the forward even
        // when all inputs are 0) we decided to keep it here.
        if input.iter().all(|x| *x == 0) {
            return 0;
        }

        let mut ops = vec![false; 7];

        // Things to do only if an actual fault is present
        if let Some(a_f) = actual_fault {
            // To avoid a great number of repetition of this match
            if let Component::Inside(real_comp) = &a_f.component {
                match real_comp {
                    LifSpecificComponent::Adder => ops[0] = true,
                    LifSpecificComponent::Multiplier => ops[1] = true,
                    LifSpecificComponent::Comparator => ops[2] = true,
                    LifSpecificComponent::Threshold => ops[3] = true,
                    LifSpecificComponent::Membrane => ops[4] = true,
                    LifSpecificComponent::Rest => ops[5] = true,
                    LifSpecificComponent::Divider => ops[6] = true,
                }
            }

            // Apply faults to v_mem, this is done every iteration because v_mem changes everytime we compute the lif formula.
            self.v_mem = apply_fault(self.v_mem, actual_fault, ops[4]);

            // This broken variable check is done to avoid doing function calls every iteration for Stuck at-X faults.
            if !self.broken {
                self.v_rest = apply_fault(self.v_rest, actual_fault, ops[5]);
                self.v_th = apply_fault(self.v_th, actual_fault, ops[3]);
                self.broken = true;
            }
        }

        // Input impulses summation
        let n_neuron = self.id as usize;
        let summation = match states_weights {
            Some(states_weights) => {
                // Actually this check is useful when we manually do a forward on a neuron passing arguments by hand,
                // that should happen only in tests, but to be sure a check is done to avoid unexpected behaviors.
                if states.len() != states_weights[n_neuron].len() {
                    panic!("Unexpected error, states array length was expected to be consistent with states_weights length");
                }

                LifNeuron::y(
                    input,
                    states,
                    &weights[n_neuron],
                    Some(&states_weights[n_neuron]),
                    actual_fault,
                    &ops,
                )
            },
            None => LifNeuron::y(input, states, &weights[n_neuron], None, actual_fault, &ops),
        };

        let exponent: f64 = div(-self.t_s_last, self.tau, actual_fault, ops[6]);
        
        // rest + (mem - rest) * exp(dt/tau) + sum(w*x -wi*xi)
        // Operation:
        // self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + summation;
        // Split in sub part, to possibly inject fault in each operation
        self.v_mem = add(
            add(
                mul(
                    add(self.v_mem, -self.v_rest, actual_fault, ops[0]),
                    exponent.exp(),
                    actual_fault,
                    ops[1],
                ),
                self.v_rest,
                actual_fault,
                ops[0],
            ),
            summation,
            actual_fault,
            ops[0],
        );

        let spike = compare(self.v_mem, self.v_th, actual_fault, ops[2]); // if v_mem>v_th then spike=1 else spike=0
        
        if spike == 1 {
            self.t_s_last = 0.0;
            self.v_mem = match self.r_type {
                ResetMode::Zero => 0.0,
                ResetMode::RestingPotential => self.v_rest,
                ResetMode::SubThreshold => add(self.v_mem, -self.v_th, actual_fault, ops[0]),
            }
        }

        // Final things to do only if an actual fault is present
        if let Some(a_f) = actual_fault {
            // Corrupting v_mem memory when the value is written back to memory
            self.v_mem = apply_fault(self.v_mem, actual_fault, ops[4]);

            // Reapplying bit flip to v_th and v_rest, only if the fault was a bit flip, because this kind of fault should happen only this time
            if let FaultType::TransientBitFlip = &a_f.fault_type {
                self.v_th = apply_fault(self.v_th, actual_fault, ops[3]);
                self.v_rest = apply_fault(self.v_rest, actual_fault, ops[5]);
            }
        }

        spike
    }
}
