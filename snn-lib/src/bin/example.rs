extern crate snn_lib;

use snn_lib::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
use snn_lib::snn::matrix_g::MatrixG;
use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
use snn_lib::snn::snn::Snn;

fn main() {
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

    // Setting personalized weights (all equal) - ONLY FOR DEBUG PURPOSES
    let mut personalized_weights = Vec::new();
    for (idx, l) in layers.iter().enumerate() {
        let mut v = Vec::new();
        for _ in 0..*l {
            if idx == 0 {
                v.push(vec![0.40; n_inputs]);
            } else {
                v.push(vec![0.40; layers[idx - 1] as usize]);
            }
        }
        personalized_weights.push(MatrixG::from(v));
    }

    // Setting personalized inner weights (all equal) - ONLY FOR DEBUG PURPOSES
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

    //let mut snn = Snn::<LifNeuron>::new(
    //    n_inputs as u32,
    //    layers,
    //    layers_inner_connections,
    //    None,
    //    None,
    //    None,
    //    seed,
    //);

    //Testing snn utility functions.
    //WORKING!
    //snn.set_neuron_parameters(&(neuron_parameters_per_layer[0].clone()), None);
    //snn.set_neuron_parameters(&neuron_parameters_per_layer[0], Some(vec![1usize]));
    //println!("{:?}", snn.get_neuron_parameters(0,0));
    //println!("{:?}", snn.get_layer_weights(0));
    //println!("{:?}", snn.get_layer_states_weights(0));
    //snn.set_layer_states_weights(Some(MatrixG::<f64>::zeroes(20,20)), 0);
    //println!("{:?}", snn.get_layer_states_weights(0));
    //snn.set_layer_weights(MatrixG::<f64>::zeroes(20,10), 0);
    //println!("{:?}", snn.get_layer_weights(0));

    // Fault injection
    let fault_configuration = FaultConfiguration::new(
        vec![
            Component::Inside(LifSpecificComponent::Adder),
            Component::Inside(LifSpecificComponent::Divider),
            Component::Inside(LifSpecificComponent::Multiplier),
            Component::Inside(LifSpecificComponent::Comparator),
            Component::Inside(LifSpecificComponent::Membrane),
            Component::Inside(LifSpecificComponent::Rest),
            Component::Inside(LifSpecificComponent::Threshold),
            Component::Outside(OuterComponent::Weights),
            Component::Outside(OuterComponent::InnerWeights),
            Component::Outside(OuterComponent::Connections),
        ],
        8,
        //FaultType::StuckAtZero,
        //FaultType::StuckAtOne,
        FaultType::TransientBitFlip,
        100,
    );

    // To start the n_occurrences faults emulations
    snn.emulate_fault(&input_matrix, &fault_configuration, log_level, None);
}
