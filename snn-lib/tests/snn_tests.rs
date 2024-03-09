#[cfg(test)]
mod snn_tests {
    use snn_lib::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
    use snn_lib::snn::layer::Layer;
    use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
    use snn_lib::snn::matrix_g::MatrixG;
    use snn_lib::snn::snn::Snn;

    //Test on snn creation
    //Consider to test with Seed

    #[test]
    fn snn_creation() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

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
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_th,
            neuron_parameters_per_layer[0].v_th
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_rest,
            neuron_parameters_per_layer[0].v_rest
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).tau,
            neuron_parameters_per_layer[0].tau
        );
        if let ResetMode::Zero = snn.get_neuron_parameters(0, 0).r_type {
            assert!(true);
        } else {
            assert!(false)
        }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);

        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(
            snn.get_layer_states_weights(0).clone().unwrap().data,
            personalized_inner_weights[0].clone().unwrap().data
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_weight_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

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
        for (idx, _) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..5 {
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
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_th,
            neuron_parameters_per_layer[0].v_th
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_rest,
            neuron_parameters_per_layer[0].v_rest
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).tau,
            neuron_parameters_per_layer[0].tau
        );
        if let ResetMode::Zero = snn.get_neuron_parameters(0, 0).r_type {
            assert!(true);
        } else {
            assert!(false)
        }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);

        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(
            snn.get_layer_states_weights(0).clone().unwrap().data,
            personalized_inner_weights[0].clone().unwrap().data
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_state_weight_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

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
        for _ in layers.iter() {
            personalized_inner_weights.push(Some(MatrixG::random(
                3usize, 5usize, true, seed, -0.40, -0.20,
            )));
        }

        // Snn creation
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_th,
            neuron_parameters_per_layer[0].v_th
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).v_rest,
            neuron_parameters_per_layer[0].v_rest
        );
        assert_eq!(
            snn.get_neuron_parameters(0, 0).tau,
            neuron_parameters_per_layer[0].tau
        );
        if let ResetMode::Zero = snn.get_neuron_parameters(0, 0).r_type {
            assert!(true);
        } else {
            assert!(false)
        }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);

        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(
            snn.get_layer_states_weights(0).clone().unwrap().data,
            personalized_inner_weights[0].clone().unwrap().data
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_layer_with_0_neurons_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let _snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_with_0_inputs_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 0;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let _snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_0_layers_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![];
        let layers_inner_connections = vec![true; layers.len()];

        let _snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_inner_connection_vector_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len() - 1];

        let _snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    fn snn_creation_with_neuron_per_layer() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );

        //We check the shape and if the weights are between 0.0 and 0.1 (hardcoded random function)
        for (i, l) in layers.iter().enumerate() {
            if i == 0 {
                assert_eq!(snn.get_layer_weights(i).cols, n_inputs);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);
            } else {
                assert_eq!(snn.get_layer_weights(i).cols, layers[i - 1] as usize);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);
            }
            assert_eq!(
                snn.get_layer_states_weights(i).clone().unwrap().rows,
                *l as usize
            );
            assert_eq!(
                snn.get_layer_states_weights(i).clone().unwrap().cols,
                *l as usize
            );
        }
    }

    #[test]
    fn snn_creation_with_trait_from() {
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];

        let mut vec: Vec<Layer<LifNeuron>> = Vec::with_capacity(layers.len());
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            vec.push(Layer::new(idx as u32, *l, None, MatrixG::from(v), None));
        }

        let snn = Snn::from(vec);
        for (i, l) in layers.iter().enumerate() {
            if i == 0 {
                assert_eq!(snn.get_layer_weights(i).cols, n_inputs);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);
            } else {
                assert_eq!(snn.get_layer_weights(i).cols, layers[i - 1] as usize);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);
            }
        }
    }

    #[test]
    #[should_panic(
        expected = "Invalid param in layer 2, weights shape expected to be [5, 3] but got [5, 10] instead"
    )]
    fn snn_creation_with_trait_from_with_wrong_weight_matrix_shape() {
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];

        let mut vec: Vec<Layer<LifNeuron>> = Vec::with_capacity(layers.len());
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            let x = if idx == 1 { 3 } else { *l };
            for _ in 0..x {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            vec.push(Layer::new(idx as u32, x, None, MatrixG::from(v), None));
        }

        let _snn = Snn::from(vec);
    }

    //FORWARD TEST
    #[test]
    fn snn_forward() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];

        let mut vec: Vec<Layer<LifNeuron>> = Vec::with_capacity(layers.len());
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            vec.push(Layer::new(idx as u32, *l, None, MatrixG::from(v), None));
        }

        let mut snn = Snn::from(vec);

        let input_matrix = MatrixG::random(24, n_inputs, false, seed, 0, 1);
        let output = snn.forward(&input_matrix, None, 0, None);
        for i in 0..output.len() {
            assert_eq!(output[i].len(), layers[layers.len() - 1] as usize);
        }
        assert_eq!(output.len(), input_matrix.rows);
    }

    //TEST ON ACTUAL FAULT?
    #[test]
    fn snn_forward_with_fault() {
        let seed = Some(42);
        let n_inputs: usize = 10;
        let layers = vec![30, 20, 10];
        let layers_inner_connections = vec![false; layers.len()];
        let fault_conf = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Inside(LifSpecificComponent::Multiplier),
                Component::Outside(OuterComponent::Connections),
            ],
            8,
            FaultType::StuckAtOne,
            100,
        );

        let mut snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );

        let input_matrix = MatrixG::random(100, n_inputs, false, seed, 0, 1);

        let output = snn.forward(&input_matrix, None, 0, None);
        let f_output = snn.forward(&input_matrix, Some(&fault_conf), 0, None);
        for i in 0..output.len() {
            assert_eq!(output[i], f_output[i]);
        }
    }
    //Since it's hard that the fault spreads through the whole network, we won't do further tests on this.

    #[test]
    fn snn_forward_output_expected() {
        let seed = Some(21);
        let n_inputs: usize = 2;
        let layers = vec![2, 1];

        let mut vec: Vec<Layer<LifNeuron>> = Vec::with_capacity(layers.len());
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![1.0; n_inputs]);
                } else {
                    v.push(vec![1.0; layers[idx - 1] as usize]);
                }
            }
            vec.push(Layer::new(idx as u32, *l, None, MatrixG::from(v), None));
        }

        let mut snn = Snn::from(vec);

        let input_matrix = MatrixG::from(vec![vec![0,0], vec![1,1]]);
        let output = snn.forward(&input_matrix, None, 0, None);
        assert_eq!(output, vec![[0],[1]]);
    }

}
