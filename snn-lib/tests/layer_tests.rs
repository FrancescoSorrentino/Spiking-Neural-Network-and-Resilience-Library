#[cfg(test)]
mod layer_tests {
    use snn_lib::snn::faults::{ActualFault, Component, FaultType, OuterComponent};
    use snn_lib::snn::layer::Layer;
    use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, ResetMode};
    use snn_lib::snn::matrix_g::MatrixG;

    #[test]
    #[should_panic(
        expected = "Invalid param, expected Weights matrix shape to be [3 , 2], but got [4, 2] instead"
    )]
    fn layer_creation_wrong_weights() {
        let weights = MatrixG::random(4, 2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3, 3, true, None, 0.01, 0.99);

        let _: Layer<LifNeuron> =
            Layer::new(0, 3, Some(states_weights.clone()), weights.clone(), None);
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected States Weights matrix shape to be [3 , 3], but got [3, 4] instead"
    )]
    fn layer_creation_wrong_states_weights() {
        let weights = MatrixG::random(3, 2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3, 4, true, None, 0.01, 0.99);

        let _: Layer<LifNeuron> =
            Layer::new(0, 3, Some(states_weights.clone()), weights.clone(), None);
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, the diagonal of the States Weights matrix must be 0.0, but got 0.08278616042737262 instead"
    )]
    fn layer_creation_wrong_states_weights_diag() {
        let weights = MatrixG::random(3, 2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3, 3, false, Some(21), 0.01, 0.99);

        let _: Layer<LifNeuron> =
            Layer::new(0, 3, Some(states_weights.clone()), weights.clone(), None);
    }

    #[test]
    fn layer_creation_and_getters() {
        let weights = MatrixG::random(3, 2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3, 3, true, None, 0.01, 0.99);

        let l: Layer<LifNeuron> = Layer::new(
            0,
            3,
            Some(states_weights.clone()),
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.0,
                v_th: 0.8,
                r_type: ResetMode::Zero,
                tau: 0.35,
            }),
        );

        // Getters tests
        assert_eq!(l.get_id(), 0);
        let neuron_params = l.get_neuron_parameters(0);
        assert_eq!(neuron_params.v_rest, 0.0);
        assert_eq!(neuron_params.v_th, 0.8);
        assert!(matches!(neuron_params.r_type, ResetMode::Zero));
        assert_eq!(neuron_params.tau, 0.35);
        assert_eq!(l.get_weights().data, weights.data);
        assert_eq!(
            l.get_states_weights().clone().unwrap().data,
            states_weights.data
        );
        assert_eq!(l.get_n_neurons(), 3);
        assert!(l.has_states_weights());
    }

    #[test]
    fn layer_setters() {
        let weights = MatrixG::random(3, 2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3, 3, true, None, 0.01, 0.99);
        let mut l: Layer<LifNeuron> = Layer::new(
            0,
            3,
            Some(states_weights.clone()),
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.0,
                v_th: 0.8,
                r_type: ResetMode::Zero,
                tau: 0.35,
            }),
        );

        // Setters tests
        let new_weights = MatrixG::random(3, 4, false, None, 0.20, 0.40);
        let new_states_weights = MatrixG::random(3, 3, true, None, 0.30, 0.60);
        l.set_weights(new_weights.clone());
        assert_eq!(l.get_weights().data, new_weights.data);
        l.set_states_weights(Some(new_states_weights.clone()));
        assert_eq!(
            l.get_states_weights().clone().unwrap().data,
            new_states_weights.data
        );
        l.set_states_weights(None);
        assert!(l.get_states_weights().is_none());
        let new_lif_params = LifNeuronParameters {
            v_rest: 0.2,
            v_th: 0.6,
            r_type: ResetMode::RestingPotential,
            tau: 0.44,
        };
        l.set_neuron_parameters(&new_lif_params);
        let neuron_params = l.get_neuron_parameters(0);
        assert_eq!(neuron_params.v_rest, 0.2);
        assert_eq!(neuron_params.v_th, 0.6);
        assert!(matches!(neuron_params.r_type, ResetMode::RestingPotential));
        assert_eq!(neuron_params.tau, 0.44);
    }

    #[test]
    fn layer_forward_with_connections_fault_stuck_at_one() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (0, Some(0)),
            FaultType::StuckAtOne,
            None,
            Some(0),
            52,
            2,
        );

        let weights = MatrixG::from(vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]]);
        let states_weights = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            Some(states_weights.clone()),
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.2,
                v_th: 2.1,
                r_type: ResetMode::RestingPotential,
                tau: 1e-5,
            }),
        );
        let mut l2 = l.clone();
        // Without the fault it does spike
        //assert_eq!(l.forward(&vec![1,0,0], Some(&af), 0), [0,0]);
        let out = l.forward(&vec![1, 0, 1], None, 0);
        let out_fault = l2.forward(&vec![1, 0, 1], Some(&af), 0);
        assert_ne!(out, out_fault)
    }

    #[test]
    fn layer_forward_with_connections_fault_stuck_at_zero() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (0, Some(0)),
            FaultType::StuckAtZero,
            None,
            Some(0),
            52, //with a 0 in the 52th the value is /2
            3,
        );

        let weights = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            None,
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.9,
                v_th: 3.0008,
                r_type: ResetMode::RestingPotential,
                tau: 1.0,
            }),
        );

        // Without the fault they do spike [1, 1]
        let out = l.forward(&vec![1, 1, 1], None, 0);

        let mut l2: Layer<LifNeuron> = Layer::new(
            1,
            2,
            None,
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.9,
                v_th: 2.5009,
                r_type: ResetMode::RestingPotential,
                tau: 1.0,
            }),
        );
        // With the fault the first neuron does not spike [0, 1]
        let out_fault = l2.forward(&vec![1, 1, 1], Some(&af), 0);

        assert_ne!(out, out_fault);
    }

    #[test]
    fn layer_forward_with_weights_fault_stuck_at_zero() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::Weights),
            1,
            (0, Some(0)),
            FaultType::StuckAtZero,
            None,
            None,
            52, //with a 0 in the 52th the value is /2
            14,
        );

        let weights = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            None,
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.9,
                v_th: 3.0008,
                r_type: ResetMode::RestingPotential,
                tau: 1.0,
            }),
        );

        // Without the fault they do spike [1, 1]
        let out = l.forward(&vec![1, 1, 1], None, 0);

        let mut l2: Layer<LifNeuron> = Layer::new(
            1,
            2,
            None,
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.9,
                v_th: 2.5009,
                r_type: ResetMode::RestingPotential,
                tau: 1.0,
            }),
        );
        // With the fault the first neuron does not spike [0, 1]
        let out_fault = l2.forward(&vec![1, 1, 1], Some(&af), 0);

        assert_ne!(out, out_fault);
    }

    #[test]
    fn layer_forward_with_weights_fault_stuck_at_one() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::Weights),
            1,
            (0, Some(0)),
            FaultType::StuckAtOne,
            None,
            Some(0),
            52,
            2,
        );

        let weights = MatrixG::from(vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]]);
        //let states_weights = MatrixG::from(vec![vec![0.0,1.0], vec![1.0,0.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            None,
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.2,
                v_th: 2.1,
                r_type: ResetMode::RestingPotential,
                tau: 1e-5,
            }),
        );
        let mut l2 = l.clone();
        let out = l.forward(&vec![1, 1, 1], None, 0);
        let out_fault = l2.forward(&vec![1, 1, 1], Some(&af), 0);

        assert_ne!(out, out_fault);
    }

    #[test]
    fn layer_forward_with_inner_weights_fault_stuck_at_zero() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::InnerWeights),
            1,
            (0, Some(1)),
            FaultType::StuckAtZero,
            None,
            None,
            63, //with a 0 in the 52th the value is /2
            14,
        );

        let weights = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let states_weights = MatrixG::from(vec![vec![0.0, -1.0], vec![-1.0, 0.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            Some(states_weights.clone()),
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.9,
                v_th: 3.0008,
                r_type: ResetMode::RestingPotential,
                tau: 1.0,
            }),
        );

        // Without the fault they do spike [1, 1]
        let out = l.forward(&vec![1, 1, 1], None, 0);
        // With the fault the first neuron does not spike [0, 1]
        let out_fault = l.forward(&vec![1, 1, 1], Some(&af), 0);
        assert_ne!(out, out_fault);
    }

    #[test]
    fn layer_forward_with_inner_weights_fault_stuck_at_one() {
        let af = ActualFault::new(
            Component::Outside(OuterComponent::InnerWeights),
            1,
            (0, Some(1)),
            FaultType::StuckAtOne,
            None,
            Some(0),
            52, //with a 0 in the 52th the value is /2
            2,
        );

        let weights = MatrixG::from(vec![vec![1., 1., 1.], vec![1., 1., 1.]]);
        let states_weights = MatrixG::from(vec![vec![0.0, -0.5], vec![-0.5, 0.0]]);
        let mut l: Layer<LifNeuron> = Layer::new(
            1,
            2,
            Some(states_weights.clone()),
            weights.clone(),
            Some(&LifNeuronParameters {
                v_rest: 0.2,
                v_th: 2.1,
                r_type: ResetMode::RestingPotential,
                tau: 1e-5,
            }),
        );
        let mut l2 = l.clone();
        // Without the fault, at the second forward it doesn't spike
        let mut _out = l.forward(&vec![1, 0, 1], None, 0);
        _out = l.forward(&vec![1, 0, 1], None, 0);
        // With the fault, it spikes at both forward.
        let mut _out_fault = l2.forward(&vec![1, 1, 1], Some(&af), 0);
        _out_fault = l2.forward(&vec![1, 1, 1], Some(&af), 0);

        assert_ne!(_out, _out_fault);
    }
}
