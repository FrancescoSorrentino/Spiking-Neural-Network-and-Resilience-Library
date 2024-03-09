extern crate snn_lib;

use snn_lib::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
use snn_lib::snn::matrix_g::MatrixG;
use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
use snn_lib::snn::snn::Snn;
fn main() {
    // Regular forward demo
    demo1();
    // Emulate faults demo
    demo2();
    // Regular forward with random weights and inner weights demo
    demo3();
}

// Regular forward demo
fn demo1() {
    // Set the seed for input matrix, weights, and inner weights
    let seed = Some(21);

    // Configuring the Snn
    let n_inputs: usize = 10;
    let layers = vec![20, 10, 5, 3];
    let layers_inner_connections = vec![true; layers.len()];

    // Setting the log level
    let log_level = 0;

    // Randomly creating an input matrix
    let input_matrix = MatrixG::random(24, n_inputs, false, seed, 0, 1);

    // 1 - First way to set specific neuron parameters, different for each layer
    let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
    for _ in 0..layers.len() {
        neuron_parameters_per_layer.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 1.5,
            r_type: ResetMode::Zero,
            tau: 0.5,
        })
    }

    // Setting personalized weights (all equal)
    let mut personalized_weights = Vec::new();
    for (idx, l) in layers.iter().enumerate() {
        let mut v = Vec::new();
        for _ in 0..*l {
            if idx == 0 {
                v.push(vec![0.60; n_inputs]);
            } else {
                v.push(vec![0.60; layers[idx - 1] as usize]);
            }
        }
        personalized_weights.push(MatrixG::from(v));
    }

    // Setting personalized inner weights (all equal)
    let mut personalized_inner_weights = Vec::new();
    for l in layers.iter() {
        personalized_inner_weights.push(Some(MatrixG::random(
            *l as usize,
            *l as usize,
            true,
            seed,
            -0.40,
            -0.20,
        )));
    }

    // Snn creation
    let mut snn = Snn::<LifNeuron>::new(
        n_inputs as u32,
        layers,
        layers_inner_connections,
        Some(neuron_parameters_per_layer.clone()),
        Some(personalized_weights),
        Some(personalized_inner_weights),
        seed,
    );

    println!("-------------Demo1:\nResult:\n{:?}\n-------------", snn.forward(&input_matrix, None, log_level, None));
}

// Emulation of faults demo
fn demo2() {
    // Set the seed for input matrix, weights, and inner weights
    let seed = Some(21);

    // Configuring the Snn
    let n_inputs: usize = 10;
    let layers = vec![20, 10, 5, 3];
    let layers_inner_connections = vec![true; layers.len()];

    // Setting the log level
    let log_level = 3;

    // Randomly creating an input matrix
    let input_matrix = MatrixG::random(24, n_inputs, false, seed, 0, 1);

    // 1 - First way to set specific neuron parameters, different for each layer
    let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
    for _ in 0..layers.len() {
        neuron_parameters_per_layer.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 1.5,
            r_type: ResetMode::Zero,
            tau: 0.5,
        })
    }

    // Setting personalized weights (all equal)
    let mut personalized_weights = Vec::new();
    for (idx, l) in layers.iter().enumerate() {
        let mut v = Vec::new();
        for _ in 0..*l {
            if idx == 0 {
                v.push(vec![0.60; n_inputs]);
            } else {
                v.push(vec![0.60; layers[idx - 1] as usize]);
            }
        }
        personalized_weights.push(MatrixG::from(v));
    }

    // Setting personalized inner weights (all equal)
    let mut personalized_inner_weights = Vec::new();
    for l in layers.iter() {
        personalized_inner_weights.push(Some(MatrixG::random(
            *l as usize,
            *l as usize,
            true,
            seed,
            -0.40,
            -0.20,
        )));
    }

    // Snn creation
    let mut snn = Snn::<LifNeuron>::new(
        n_inputs as u32,
        layers,
        layers_inner_connections,
        Some(neuron_parameters_per_layer.clone()),
        Some(personalized_weights),
        Some(personalized_inner_weights),
        seed,
    );

    // Fault injection
    let fault_configuration = FaultConfiguration::new(
        vec![
            Component::Inside(LifSpecificComponent::Comparator),
            Component::Outside(OuterComponent::Weights)
        ],
        8,
        FaultType::StuckAtZero,
        100,
    );

    // To start the n_occurrences faults emulations
    snn.emulate_fault(&input_matrix, &fault_configuration, log_level, None);
}


// Regular forward with random weights and inner weights demo
fn demo3() {
    // Set the seed for input matrix, weights, and inner weights
    let seed = Some(21);

    // Configuring the Snn
    let n_inputs: usize = 10;
    let layers = vec![20, 10, 5, 3];
    let layers_inner_connections = vec![true; layers.len()];

    // Setting the log level
    let log_level = 0;

    // Randomly creating an input matrix
    let input_matrix = MatrixG::random(24, n_inputs, false, seed, 0, 1);

    // 1 - First way to set specific neuron parameters, different for each layer
    let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
    for _ in 0..layers.len() {
        neuron_parameters_per_layer.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 1.5,
            r_type: ResetMode::Zero,
            tau: 0.5,
        })
    }

    // Snn creation
    let mut snn = Snn::<LifNeuron>::new(
        n_inputs as u32,
        layers,
        layers_inner_connections,
        Some(neuron_parameters_per_layer.clone()),
        None,
        None,
        seed,
    );

    println!("-------------Demo3:\nResult:\n{:?}\n-------------", snn.forward(&input_matrix, None, log_level, None));
}