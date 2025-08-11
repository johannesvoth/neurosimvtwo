from __future__ import annotations

from typing import List

from network import (
    Network,
    Neuron,
    Connection,
    PixelInputNeuron,
    make_pixel_input_neurons,
    show_random_binary_image,
)
from simulator import Simulator, SimulatorConfig


def build_demo_network(num_inputs: int = 8) -> Network:
    input_neurons: List[PixelInputNeuron] = make_pixel_input_neurons(start_id=0, count=num_inputs, i_on=10.0)

    n_hidden1 = Neuron.fast_spiking(neuron_id=num_inputs)
    n_hidden2 = Neuron.regular_spiking(neuron_id=num_inputs + 1)

    neurons: List[Neuron] = [*input_neurons, n_hidden1, n_hidden2]

    connections: List[Connection] = []
    #for inp in input_neurons:
     #   connections.append(Connection(source_id=inp.id, target_id=n_hidden1.id, weight=1.0))
      #  connections.append(Connection(source_id=inp.id, target_id=n_hidden2.id, weight=1.0))

    return Network(neurons=neurons, connections=connections)


def main() -> None:
    net = build_demo_network(num_inputs=8)
    sim = Simulator(model=net, config=SimulatorConfig(dt_ms=1.0, input_current=0.0, synaptic_scale=20.0))

    print("Initialized network:")
    for n in net.neurons:
        print(f"  neuron {n.id}: v={n.v:.1f}, u={n.u:.1f}, a={n.a}, b={n.b}, c={n.c}, d={n.d}")

    # Input neurons will be driven by random binary images
    input_ids = [n.id for n in net.neurons if isinstance(n, PixelInputNeuron)]
    print(f"Input neuron IDs: {input_ids}")

    print("\nSimulating (spikes shown as *):")
    steps = 150
    image_interval = 100  # show a new image every 100 steps
    for t in range(steps):
        if t % image_interval == 0:
            inputs = [n for n in net.neurons if isinstance(n, PixelInputNeuron)]
            pattern = show_random_binary_image(inputs, white_probability=0.5)
            print(f"t={t:04d} new image: {pattern}")
        sim.step()
        spikes_symbols = ["*" if n.spiked else "." for n in net.neurons]
        voltages = [round(n.v, 1) for n in net.neurons]
        print(f"t={t:04d}  spikes={' '.join(spikes_symbols)}  v={voltages}")


if __name__ == "__main__":
    main()


