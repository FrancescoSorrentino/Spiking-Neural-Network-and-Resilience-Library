# Group45

# Spiking Neural Network + Resilience Library
## snn-lib

- [Members](#members)
- [Description](#description)
- [Main structures](#main-structures)
- [Main methods](#main-methods)
- [Network parameters setting](#network-parameters-setting)
- [Usage examples](#usage-examples)
- [Licence](#licence)

## Members
- Francesco Sorrentino (s301665)
- Marco Sacchet (s295033)
- Simone Mascali (s305001)

## Description
This repository contains a Rust library for the study of the resilience of a Spiking Neural Network in terms of hardware faults that can happen during inference.

### Hardware Simulation
This library emulates a physical implementation of a Spiking Neural Network. The simulation is based on a few assumptions:
- The Neural Network works with 64bit weights, so it's not intended to be used with quantized networks.
- Each neuron is physically mapped, so we assume that each neuron has its own process unit (with pre-defined components: i.e: Adder, Multiplier, etc...) and its own associated memory which contains the internal information of the neuron (i.e: membrane value, threshold value, etc...)
- Layers weights and inner weights are stored in a specific memory area (we suppose main memory) and are transferred through "n" buses (where n is specified in the fault simulation) that can transfer a whole weight. The previous assumption means that if with 64 bits weights, buses will be 64 bits. 
 
## Main structures

### Snn
Snn is the main structure that contains the whole network, made as an array of Layers of a generic type `N` that represents the neuron type.

```rust
pub struct Snn<N: Neuron> {
layers: Vec<Layer<N>>,
}
```

### Layer
Each layer has an id, an array of neurons, the output of the layer at the previous instant (`states`), weights relative to the connections between the previous layer and this one, 
weights of connections between neurons of the same layer (`states_weights`).

```rust
pub struct Layer<N: Neuron> {
    id: u32,
    neurons: Vec<N>,
    states: Vec<u8>,
    weights: MatrixG<f64>,
    states_weights: Option<MatrixG<f64>>,
}
```

### LifNeuron
The LifNeuron is a specific neuron implementation of the generic trait `Neuron`. \
It has an id, a membrane voltage value, a resting potential, a voltage threshold to evaluate when the neuron spikes, a reset mode to specify how the membrane voltage value will be reset after a spike, a time accumulator of time steps to 
keep track of time passing, the time constant tau, the time step size and an information about the neuron to be broken or not (because some faults have to be applied just once
at the first forward call).

```rust
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
```

### FaultConfiguration
The FaultConfiguration structure is the one given by the user to tell which faults should be applied, with which characteristics and how many times the fault emulation have to be done.

```rust
pub struct FaultConfiguration<D: SpecificComponent + Clone + Debug> {
    components: Vec<Component<D>>,
    n_bus: usize,
    fault_type: FaultType,
    n_occurrences: u32,
}
```

## Main methods
### Emulate fault
The emulate fault function takes the input matrix as input, the fault configuration given by the user, the desired [log level](#log-level-setting) 
and an optional seed for the random generation of the actual faults (decide which layer to break, which neuron, etc...). 

```rust
fn emulate_fault(&mut self, input_matrix: &MatrixG<u8>, fault_configuration: &FaultConfiguration<N::D>, log_level: u8, seed: Option<u64>) -> ()
```

### Forward (Snn)
The forward function of the SNN takes the same arguments as the [emulate fault](#emulate-fault).
Creates the channels for thread communication, to make the snn working as a pipeline. 
The actual fault is generated here and is given to the forward function of the layer
involved in the fault. \
Layers are distributed between threads, to use the maximum number of threads with the minimum possible overhead.

```rust
fn forward(&mut self, input_matrix: &MatrixG<u8>, fault_configuration: Option<&FaultConfiguration<N::D>>, log_level: u8, seed: Option<u64>) -> Vec<Vec<u8>>
```

### Forward (Layer)
The Layer struct has its own forward function. It takes as arguments the vector of inputs (for the layers after the first one it's the output of the previous layer), 
the specific fault for that specific occurrence and, finally, time: since it is a spike snn, the input of the snn is mapped with shape (time, features). In our case, time is needed to
know when the fault has to be applied.

```rust
fn forward(&mut self, inputs: &Vec<u8>, actual_fault: Option<&ActualFault<N::D>>, time: usize) -> Vec<u8>
```
### Forward (LifNeuron)
The forward of the LifNeuron it's just an implementation of the forward in Trait `Neuron`.
Inside it's a scalar product between input, weights and, if present, internal weights (`state_weights`). \
`actual_fault` contains information about possible fault happening to the internal components of the neuron.
The other parameters used for the neuron internal computation are stored within the neuron.

```rust
fn forward(&mut self, input: &Vec<u8>, states_weights: &Option<MatrixG<f64>>, weights: &MatrixG<f64>, states: &Vec<u8>, actual_fault: Option<&ActualFault<LifSpecificComponent>>) -> u8 
```


## Network parameters setting
### Seed setting
To reproduce a certain behavior or check the difference of outputs with a specific network setting, 
a specific seed can be set and passed to the proper function for the random generations used inside the network parameters setting.\
The seed can be set for the generation of: input matrix, weights and inner weights.\
Note that it should be an `Option<u64>`, if None is given then the default setting for the PRNG (Pseudo Random Number Generator) is used.

```rust
// Set the seed for input matrix, weights, and inner weights 
let seed = Some(21);
// ...
// Passing the seed here set that seed for the input matrix generation
let input_matrix = MatrixG::random(rows, cols, false, seed, 0, 1);
// AND/OR
// Passing the seed here set that seed for weights and inner weights generation.
let mut snn = Snn::<LifNeuron>::new(..., seed);
```

### Log level setting
Log level can be selected between these values:
- 0: print only the results of emulation on `stdout`, without any details about the network's structure or the injected faults
- 1: print on `stdout` network structure, requested injected fault, input matrix and the applied fault for every result.
- 2: print only the results of emulation on a log file, without any details about the network's structure or the injected faults
- 3: print on a log file network structure, requested injected fault, input matrix and the applied fault for every result.
```rust
// Setting the log level
let log_level = 0;
```

### Basic configuration of the SNN
You must set the `n_inputs` as the number of neurons in the input layer, `layers` vector as vector of neurons per inner layer, and `layers_inner_connection` as parallel information about each layer:
`true` if inner connections are present, `false` otherwise.
```rust
// Configuring the Snn
let n_inputs: usize = 20;
let layers = vec![15,10,5,2];
let layers_inner_connections = vec![true, false, true, true];
```

### Weights setting
If you want to set specific weights for the network you can create a `Vec<MatrixG>`.\
You must build each MatrixG using `MatrixG::from(Vec<Vec<T>>`) where `vec[i][j]` represents the weight connection between neuron i of layer n and neuron j of layer n-1.
```rust
let mut personalized_weights = Vec::new();
personalized_weights.push(MatrixG::from(vec![vec![...], vec![...], ...]));
// ...
let mut snn = Snn::<LifNeuron>::new(..., Some(personalized_weights), ...);
```

### Inner weights setting
If you want to set specific inner weights for the network you can create a `Vec<MatrixG>`.\
You must build each MatrixG using `MatrixG::from(Vec<Vec<T>>`) where `vec[i][j]` represents the inner weight connection between neuron i and neuron j of the same layer,
the matrix is a square matrix and the diagonal should be 0 otherwise it would represent the inner weights of a neuron with itself.
```rust
let mut personalized_inner_weights = Vec::new();
personalized_inner_weights.push(MatrixG::from(vec![vec![...], vec![...], ...]));
// ...
let mut snn = Snn::<LifNeuron>::new(..., Some(personalized_inner_weights), ...);
```

### Neuron parameters setting
There is a specific struct to define neuron specific parameters (Lif in this case):

```rust
let parameters_for_lif = LifNeuronParameters {
    v_rest: 0.0,
    v_th: 0.8,
    r_type: ResetMode::Zero,
    tau: 0.35,
};
```

And then you have multiple ways to set parameters of neurons inside the network layers:

```rust
// 1 - First way: set different neuron parameters for each layer
let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
for _ in 0..layers.len() {
    neuron_parameters_per_layer.push(LifNeuronParameters { ... })
}
let mut snn = Snn::<LifNeuron>::new(..., Some(neuron_parameters_per_layer), ...);

// 2 - Second way: set the same parameters for the whole network (same neuron parameters for all layers)
let parameters_for_all_lif = LifNeuronParameters { ... };
snn.set_neuron_parameters(parameters_for_all_lif, None);

// 3 - Third way: set the same parameters for specific layers in the network (specified by a vector of layer indexes)
let parameters_for_lif_of_some_layers = LifNeuronParameters { ... };
snn.set_neuron_parameters(parameters_for_lif_of_some_layers, Some(vec![...]));
```
### Fault configuration setting
The FaultConfiguration struct is used to specify the fault configuration that has to be emulated.\
The vector of Component represents the component that could present a fault during the emulation.\
The bus number is the number of bus and one of these bus could present a fault during the emulation.\
The fault type is an enum that specifies which type of fault has to occur during the emulation.\
The last number is the desired number of repetitions of the fault emulation. 

```rust
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
    8, //Bus number
    FaultType::TransientBitFlip, //or FaultType::StuckAtZero or FaultType::StuckAtOne 
    100,
);
```

## Usage examples
**See `bin/example.rs` for a complete example of usage.**

To create a SNN configured with all parameters explained before:

```rust
// Snn creation
let mut snn = Snn::<LifNeuron>::new(n_inputs, layers, layers_inner_connections, Some(neuron_parameters_per_layer), Some(personalized_weights), Some(personalized_inner_weights), seed);
```

To randomly create an input matrix:
```rust
 // Randomly creating an input matrix
let input_matrix = MatrixG::random(length, n_inputs, false, seed, 0, 1);
```

To run a simple network inference over the input matrix:

```rust
// To start the inference without faults
println!("The final result is: {:?}", snn.forward(&input_matrix, None, log_level, None));
```

For emulation of faults:

```rust
// Fault injection
let fault_configuration = FaultConfiguration::new(
    vec![
        Component::Inside(LifSpecificComponent::Adder),
        Component::Inside(LifSpecificComponent::Divider),
        Component::Inside(LifSpecificComponent::Threshold),
        Component::Outside(OuterComponent::Weights),
    ],
    8,
    FaultType::TransientBitFlip,
    100,
);

// To start the n_occurrences faults emulations
snn.emulate_fault(&input_matrix, &fault_configuration, 1);
```

## Licence
Released under MIT Licence.\
Note: this is a university project in the partial fulfillment of a course exam, no guarantees on the absence of errors or bugs that may not have been found and considered yet.