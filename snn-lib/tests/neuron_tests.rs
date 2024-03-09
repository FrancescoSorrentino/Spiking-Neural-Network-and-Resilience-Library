#[cfg(test)]
mod neuron_tests {
    use snn_lib::snn::faults::{ActualFault, Component, FaultType};
    use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
    use snn_lib::snn::matrix_g::MatrixG;
    use snn_lib::snn::neuron::Neuron;

    #[test]
    fn neuron_creation() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(neuron.get_id(), 0);
        assert_eq!(neuron.get_parameters().v_rest, p.v_rest);
        assert_eq!(neuron.get_parameters().v_th, p.v_th);
        assert_eq!(neuron.get_parameters().tau, p.tau);
        assert!(matches!(
            neuron.get_parameters().r_type,
            ResetMode::RestingPotential
        ));
    }

    #[test]
    fn neuron_creation_default() {
        let neuron = LifNeuron::new(3, None);
        assert_eq!(neuron.get_id(), 3);
        assert_eq!(neuron.get_parameters().v_rest, 0.0);
        assert_eq!(neuron.get_parameters().v_th, 0.8);
        assert_eq!(neuron.get_parameters().tau, 5e-3);
        assert!(matches!(neuron.get_parameters().r_type, ResetMode::Zero));
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, v_th must be strictly positive")]
    fn neuron_creation_panic_threshold() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let _neuron = LifNeuron::new(0, Some(&p));
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, tau must be strictly positive")]
    fn neuron_creation_panic_tau() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 0.0,
        };
        let _neuron = LifNeuron::new(0, Some(&p));
    }

    #[test]
    fn neuron_setter() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters {
            v_rest: 1.0,
            v_th: 1.01,
            r_type: ResetMode::SubThreshold,
            tau: 1.1,
        };
        neuron.set_parameters(&p1);
        assert_eq!(neuron.get_parameters().v_rest, p1.v_rest);
        assert_eq!(neuron.get_parameters().v_th, p1.v_th);
        assert_eq!(neuron.get_parameters().tau, p1.tau);
        assert!(matches!(
            neuron.get_parameters().r_type,
            ResetMode::SubThreshold
        ))
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, v_th must be strictly positive")]
    fn neuron_setter_wrong_v_th() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters {
            v_rest: 1.0,
            v_th: -0.2,
            r_type: ResetMode::SubThreshold,
            tau: 1.1,
        };
        neuron.set_parameters(&p1);
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, tau must be strictly positive")]
    fn neuron_setter_wrong_tau() {
        let p = LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters {
            v_rest: 1.0,
            v_th: 0.2,
            r_type: ResetMode::SubThreshold,
            tau: 0.0,
        };
        neuron.set_parameters(&p1);
    }

    // Forward checking from now on
    #[test]
    fn neuron_forward_zero_input() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![0, 0, 0], &None, &w, &vec![0, 0], None),
            0
        );
    }

    /** Manual calculation check to test the neuron forward spike or not (next few tests)
    v_rest = 0.9
    tau = 1.0
    v_th = 0.01
    time_step = 1e-3
    RestingPotential
    ------------------------------
    input = [1,1,1]
    weights = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]
    ------------------------------
    scalar (initial) = 0.0
    scalar = 0.0 + 1.0 = 1.0 => 1.0 + 1.0 = 2.0 => 2.0 + 1.0 = 3.0
        => out = 3.0
    y out = 3.0 = summation
    ------------------------------
    v_mem (initial) = 0.0
    v_mem = [ (0.0 (v_mem_initial) - 0.9 (v_rest))*exp[(-1e-3 (t_s_last) / 1.0 (tau))] ] + 0.9 (v_rest) + summation => 3.00089955
    ------------------------------
    spike = 3.00089955 > 0.01 => 1 (true)
    => t_s_last = 0 (reset the last spike time save)
    v_mem finale = v_rest = 0.9 (RestingPotential)
    */
    #[test]
    fn neuron_forward_spiking() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            1
        );
    }

    #[test]
    fn neuron_forward_spiking_2() {
        let w = MatrixG::from(vec![vec![1.0, 0.0, 1.0], vec![0.0, 0.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 0, 1], &None, &w, &vec![0, 0], None),
            1
        );
    }

    #[test]
    fn neuron_forward_spiking_3() {
        let w = MatrixG::from(vec![vec![0.52, 0.0, 0.72], vec![0.0, 0.0, 0.0]]);
        // With this weights it results in:
        // summation (scalar function) = 0.0 + 0.52 = 0.52 => 0.52 + 0.0 = 0.52 => 0.52 + 0.72 = 1.24
        // 	=> summation = 1.24
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 1.25,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        // v_mem = 1.2408995501499624 => 1.2408... < 1.25 => not spiking
        assert_eq!(
            neuron.forward(&vec![1, 0, 1], &None, &w, &vec![0, 0], None),
            0
        );
    }

    #[test]
    fn neuron_forward_not_spiking() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        // Setting the v_rest to 3.00089956 it should not spike (manually calculated to check algorithm correctness
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 3.00089956,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            0
        );
    }

    #[test]
    fn neuron_forward_almost_not_spiking() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        // Setting the v_rest to 3.00089954 it should spike (manually calculated to check algorithm correctness
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 3.00089954,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            1
        );
    }

    #[test]
    fn neuron_forward_spiking_twice() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0, 0], None),
            1
        );
        // Now v_mem should be 0.9 at the beginning, so: V_mem(t_s_-1) - V_rest = 0 => also the e^(...) is multiplied by 0
        // In the end the next forward should bring the neuron V_mem to 3.9 and so spikes again
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            1
        );
    }

    #[test]
    fn neuron_forward_spiking_once() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            1
        );
        // Now v_mem should be 0.9 at the beginning, so: V_mem(t_s_-1) - V_rest = 0 => also the e^(...) is multiplied by 0
        // => v_mem = [ (0.9 (v_mem_prec) - 0.9 (v_rest))*exp[(-1e-3 (t_s_last) / 1.0 (tau))] ] + 0.9 (v_rest) + summation = 3.9
        // So in the end the next forward should bring the neuron V_mem to 3.9 and so changing the V_th avoids the second spiking:
        let p2 = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 3.9,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        neuron.set_parameters(&p2);
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &None, &w, &vec![0, 0], None),
            0
        );
    }

    // Now checking also with states weights
    #[test]
    #[should_panic(
        expected = "Unexpected error, states array length was expected to be consistent with states_weights length"
    )]
    fn neuron_forward_states_and_states_weights_inconsistent() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 0.01,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &Some(sw), &w, &vec![0, 0, 0], None),
            1
        );
    }
    #[test]
    fn neuron_forward_with_states_weights_spiking() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        // summation (scalar function) = 0.0 + 1.0 = 1.0 => 1.0 + 1.0 = 2.0 => 2.0 + 1.0 = 3.0 =>
        // (now it starts with states weights)
        // => 3.0 + 0.0 = 3.0 => 3.0 + 1.0 => 4.0
        // 	=> summation = 4.0
        // v_mem = [ (0.0 (v_mem_prec) - 0.9 (v_rest))*exp[-1e-3 (t_s_last) / 1.0 (tau)] ] + 0.9 (v_rest) + summation => 4.00089955 that is greater than V_th => spike
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &Some(sw), &w, &vec![1, 1], None),
            1
        );
    }

    #[test]
    fn neuron_forward_with_states_weights_not_spiking() {
        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0009,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron = LifNeuron::new(0, Some(&p));
        // summation (scalar function) = 0.0 + 1.0 = 1.0 => 1.0 + 1.0 = 2.0 => 2.0 + 1.0 = 3.0 =>
        // (now it starts with states weights)
        // => 3.0 + 0.0 = 3.0 => 3.0 + 1.0 => 4.0
        // 	=> summation = 4.0
        // v_mem = [ (0.0 (v_mem_prec) - 0.9 (v_rest))*exp[-1e-3 (t_s_last) / 1.0 (tau)] ] + 0.9 (v_rest) + summation => 4.00089955 that is smaller than V_th => not spike
        assert_eq!(
            neuron.forward(&vec![1, 1, 1], &Some(sw), &w, &vec![1, 1], None),
            0
        );
    }

    // Now checking also with faults

    #[test]
    fn neuron_forward_with_adder_fault_stuck_at_one() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Adder),
            0,
            (0, None),
            FaultType::StuckAtOne,
            None,
            None,
            63,
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_adder_fault_stuck_at_zero() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Adder),
            0,
            (0, None),
            FaultType::StuckAtZero,
            None,
            None,
            52, //with a 0 in the 52th each value is /2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_adder_fault_stuck_at_one_2() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Adder),
            0,
            (0, None),
            FaultType::StuckAtOne,
            None,
            None,
            0, //with a 1 in the 0th each value has a far away decimal change
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it spikes anyway because a 1 in the 0th position of a f64 provokes a very slight change
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            1
        );
    }

    #[test]
    fn neuron_forward_with_adder_fault_stuck_at_one_3() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Adder),
            0,
            (0, None),
            FaultType::StuckAtOne,
            None,
            None,
            51, //with a 1 in the 51th a positive value can have a slight increment
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it spikes anyway because a positive value can have an increment
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            1
        );
    }

    #[test]
    fn neuron_forward_with_adder_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Adder),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            52, //with a 0 in the 52th each value is /2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike because each value is /2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_multiplier_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Multiplier),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            52, //with a 0 in the 52th each value is /2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike because each value is /2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_divider_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Divider),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            52, //with a 0 in the 52th each value is /2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike because each value is /2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_membrane_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Membrane),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            52, //with a 0 in the 52th each value is /2, with a 1 is *2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it spikes anyway because each value is *2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            1
        );
    }

    #[test]
    fn neuron_forward_with_rest_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Rest),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            52, //with a 0 in the 52th each value is /2, with a 1 is *2
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it spikes anyway because rest value is *2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            1
        );
    }

    #[test]
    fn neuron_forward_with_threshold_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Threshold),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            53, //with a 0 in the 53th each value is /4, with a 1 is *4
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it spikes anyway because rest value is *2
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }

    #[test]
    fn neuron_forward_with_comparator_fault_bitflip() {
        let af = ActualFault::new(
            Component::Inside(LifSpecificComponent::Comparator),
            0,
            (0, None),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            53, //with a 0 in the 53th each value is /4
            2,
        );

        let w = MatrixG::from(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let sw = MatrixG::from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let p = LifNeuronParameters {
            v_rest: 0.9,
            v_th: 4.0008,
            r_type: ResetMode::RestingPotential,
            tau: 1.0,
        };
        let mut neuron1 = LifNeuron::new(0, Some(&p));
        // Without the fault it does spike
        assert_eq!(
            neuron1.forward(&vec![1, 1, 1], &Some(sw.clone()), &w, &vec![1, 1], None),
            1
        );
        let mut neuron2 = LifNeuron::new(0, Some(&p));
        // With the fault it does not spike because the comparator with the v_th is broken and so it outputs something != 1 (0.25 in this case, rounded to 0)
        assert_eq!(
            neuron2.forward(
                &vec![1, 1, 1],
                &Some(sw.clone()),
                &w,
                &vec![1, 1],
                Some(&af)
            ),
            0
        );
    }
}
