#[cfg(test)]
mod fault_tests {
    use snn_lib::snn::faults::{
        ActualFault, Component, FaultConfiguration, FaultType, OuterComponent,
    };
    use snn_lib::snn::lif::LifSpecificComponent;

    #[test]
    fn create_fault_conf() {
        let f_c = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            8,
            FaultType::TransientBitFlip,
            100,
        );
        assert_eq!(f_c.components_contain_inner_weights(), false);
    }

    #[test]
    #[should_panic(expected = "Invalid param, expected number of components at least 1, but got 0")]
    fn create_fault_conf_panic_comp() {
        let f_c = FaultConfiguration::<LifSpecificComponent>::new(
            vec![],
            8,
            FaultType::TransientBitFlip,
            100,
        );
        assert_eq!(f_c.components_contain_inner_weights(), false);
    }

    #[test]
    #[should_panic(expected = "Invalid param, expected number of buses at least 1, but got 0")]
    fn create_fault_conf_panic_bus() {
        let _ = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            0,
            FaultType::TransientBitFlip,
            100,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected number of occurrences at least 1, but got 0"
    )]
    fn create_fault_conf_panic_occurrences() {
        let _ = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            1,
            FaultType::TransientBitFlip,
            0,
        );
    }

    #[test]
    fn fault_conf_has_inner_weight() {
        let fc = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            1,
            FaultType::TransientBitFlip,
            5,
        );
        assert_eq!(fc.components_contain_inner_weights(), false);

        let fc = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
                Component::Outside(OuterComponent::InnerWeights),
            ],
            1,
            FaultType::TransientBitFlip,
            5,
        );
        assert_eq!(fc.components_contain_inner_weights(), true);
    }

    #[test]
    fn fault_conf_get_n_occurrences() {
        let fc = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
                Component::Outside(OuterComponent::InnerWeights),
            ],
            1,
            FaultType::TransientBitFlip,
            5,
        );
        assert_eq!(fc.get_n_occurrences(), 5);
    }

    #[test]
    fn generate_actual_fault() {
        let f_c = FaultConfiguration::new(
            vec![Component::Inside(LifSpecificComponent::Adder)],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, false)];
        let res = f_c.get_actual_fault(layers_info, 5, None);
        assert!(matches!(
            res.component,
            Component::Inside(LifSpecificComponent::Adder)
        ));
        assert_eq!(res.layer_id, 0);
        assert!(res.neuron_id.0 < 3);
        assert!(matches!(res.fault_type, FaultType::TransientBitFlip));
        assert!(res.time_tbf.unwrap() < 5);
        assert!(res.bus.is_none());
        assert!(res.offset < 64);

        let f_c = FaultConfiguration::new(
            vec![Component::Inside(LifSpecificComponent::Adder)],
            8,
            FaultType::StuckAtOne,
            100,
        );

        let layers_info = vec![(3, false)];
        let res = f_c.get_actual_fault(layers_info, 5, None);
        assert!(matches!(
            res.component,
            Component::Inside(LifSpecificComponent::Adder)
        ));
        assert_eq!(res.layer_id, 0);
        assert!(res.neuron_id.0 < 3);
        assert!(matches!(res.fault_type, FaultType::StuckAtOne));
        assert!(res.time_tbf.is_none());
        assert!(res.bus.is_none());
        assert!(res.offset < 64);
    }

    #[test]
    #[should_panic(expected = "Invalid component, there is no Layer with Inner Weights")]
    fn generate_actual_fault_panic_inner_weight() {
        let f_c = FaultConfiguration::<LifSpecificComponent>::new(
            vec![Component::Outside(OuterComponent::InnerWeights)],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, false)];
        let _res = f_c.get_actual_fault(layers_info, 5, None);
    }

    #[test]
    #[should_panic(expected = "Invalid param, expected total time at least 1, but got 0")]
    fn generate_actual_fault_panic_if_time_zero() {
        let f_c = FaultConfiguration::<LifSpecificComponent>::new(
            vec![Component::Outside(OuterComponent::InnerWeights)],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, true)];
        let _res = f_c.get_actual_fault(layers_info, 0, None);
    }

    #[test]
    fn create_actual_fault() {
        let af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (2, Some(1)),
            FaultType::TransientBitFlip,
            Some(0),
            Some(1),
            2,
            3,
        );
        assert!(matches!(
            af.component,
            Component::Outside(OuterComponent::Connections)
        ));
        assert_eq!(af.layer_id, 1);
        assert_eq!(af.neuron_id.0, 2);
        assert_eq!(af.neuron_id.1, Some(1));
        assert!(matches!(af.fault_type, FaultType::TransientBitFlip));
        assert_eq!(af.time_tbf, Some(0));
        assert_eq!(af.bus, Some(1));
        assert_eq!(af.offset, 2);
        assert_eq!(af.get_n_bus(), 3);
    }

    #[test]
    #[should_panic(expected = "Invalid param, maximum offset is 63, but got 65")]
    fn create_actual_fault_panic_offset() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (2, Some(1)),
            FaultType::TransientBitFlip,
            Some(0),
            Some(1),
            65,
            3,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected time_tbf to be Some(usize) if fault_type is FaultType::TransientBitFlip, but got None"
    )]
    fn create_actual_fault_panic_tbf() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (2, Some(1)),
            FaultType::TransientBitFlip,
            None,
            Some(1),
            2,
            3,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, faulted bus: 6 greater than total number of buses: 3"
    )]
    fn create_actual_fault_panic_greater_bus() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (2, Some(1)),
            FaultType::TransientBitFlip,
            Some(0),
            Some(6),
            2,
            3,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected bus to be Some(usize) if component is OuterComponent::Connections, but got None"
    )]
    fn create_actual_fault_panic_no_bus() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            1,
            (2, Some(1)),
            FaultType::TransientBitFlip,
            Some(0),
            None,
            2,
            3,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected neuron_id.1 to be Some(u32) if component is Component::Outside, but got None"
    )]
    fn create_actual_fault_panic_no_id2() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Weights),
            1,
            (2, None),
            FaultType::TransientBitFlip,
            Some(0),
            Some(1),
            2,
            3,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected layer_id to be greater than 1 if component is Component::Outside, but got 0"
    )]
    fn create_actual_fault_panic_layer() {
        let _af = ActualFault::<LifSpecificComponent>::new(
            Component::Outside(OuterComponent::Connections),
            0,
            (2, Some(3)),
            FaultType::TransientBitFlip,
            Some(0),
            Some(1),
            2,
            3,
        );
    }
}
