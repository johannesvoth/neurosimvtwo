from __future__ import annotations

from typing import List

from network import (
    Network,
    Neuron,
    Connection,
    PixelInputNeuron,
    PixelOutputNeuron,
    make_pixel_input_neurons,
    make_hidden_neurons,
    fully_connect,
    make_pixel_output_neurons,
    read_output_binary_image,
    reset_all_states,
)
from simulator import Simulator, SimulatorConfig
from training import (
    simple_delta_train_step,
    squared_error,
)


def build_demo_network(num_inputs: int = 8) -> Network:
    # Tunables for hidden layer
    hidden_count = 6 # number of hidden neurons
    input_to_hidden_weight = 1.0
    hidden_recurrent_weight = 0.5
    hidden_to_output_weight = 1.0

    input_neurons: List[PixelInputNeuron] = make_pixel_input_neurons(
        start_id=0, count=num_inputs, i_on=20.0, d=0.2
    )

    hidden_neurons: List[Neuron] = make_hidden_neurons(
        start_id=num_inputs, count=hidden_count, preset="fast_spiking"
    )

    # Output neurons mirror the input size so it forms an image
    output_neurons: List[PixelOutputNeuron] = make_pixel_output_neurons(
        start_id=num_inputs + hidden_count, count=num_inputs, on_duration_steps=5
    )

    neurons: List[Neuron] = [*input_neurons, *hidden_neurons, *output_neurons]

    connections: List[Connection] = []
    # Connect inputs to all hidden
    for inp in input_neurons:
        for h in hidden_neurons:
            connections.append(
                Connection(source_id=inp.id, target_id=h.id, weight=input_to_hidden_weight)
            )
    # Fully connect hidden layer (no self-loops)
    connections += fully_connect([n.id for n in hidden_neurons], weight=hidden_recurrent_weight)

    # Connect all hidden to all outputs
    for h in hidden_neurons:
        for outp in output_neurons:
            connections.append(
                Connection(source_id=h.id, target_id=outp.id, weight=hidden_to_output_weight)
            )

    return Network(neurons=neurons, connections=connections)


def main() -> None:
    net = build_demo_network(num_inputs=8)
    sim = Simulator(model=net, config=SimulatorConfig(dt_ms=1.0, input_current=0.0, synaptic_scale=20.0))

    # Minimal usage example
    steps = 150
    # Manually coded single image for input neurons (True=white/ON, False=black/OFF)
    manual_pattern: List[bool] = [True, False, True, False, True, False, True, False]
    inputs = [n for n in net.neurons if isinstance(n, PixelInputNeuron)]
    for n, is_white in zip(inputs, manual_pattern):
        n.encode(is_white)
    print("Input image:", manual_pattern)

    # Training routine: run simple delta training for X epochs
    train_steps = 120
    delta = 0.1
    epochs = 1000

    initial_pred = read_output_binary_image([n for n in net.neurons if isinstance(n, PixelOutputNeuron)])
    initial_err = squared_error(manual_pattern, initial_pred)
    print("Initial error:", initial_err)

    for idx in range(1, epochs + 1):
        baseline_err, candidate_err, accepted = simple_delta_train_step(
            net, sim.config, manual_pattern, steps=train_steps, delta=delta
        )
        print(
            f"Epoch {idx:04d}: baseline={baseline_err:.3f} candidate={candidate_err:.3f} accepted={accepted}",
            flush=True,
        )

    # Reset state and re-encode inputs for a clean rollout after training
    reset_all_states(net)
    for n, is_white in zip(inputs, manual_pattern):
        n.encode(is_white)

    # Single evaluation cycle and final readout
    eval_steps = train_steps
    for _ in range(eval_steps):
        sim.step()
    output_neurons = [n for n in net.neurons if isinstance(n, PixelOutputNeuron)]
    final_output = read_output_binary_image(output_neurons)
    final_err = squared_error(manual_pattern, final_output)
    print("Final output image:", final_output)
    print("Final error:", final_err)


if __name__ == "__main__":
    main()


