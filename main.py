from __future__ import annotations

from typing import List

from network import Network, Neuron, Connection
from simulator import Simulator, SimulatorConfig


def build_demo_network() -> Network:
    neurons: List[Neuron] = [
        Neuron.regular_spiking(neuron_id=0),
        Neuron.fast_spiking(neuron_id=1),
        Neuron.intrinsically_bursting(neuron_id=2),
    ]

    connections = [
        Connection(source_id=0, target_id=1, weight=1.0),
        #Connection(source_id=1, target_id=2, weight=5.0),
        #Connection(source_id=0, target_id=2, weight=3.0),
    ]

    return Network(neurons=neurons, connections=connections)


def main() -> None:
    net = build_demo_network()
    sim = Simulator(model=net, config=SimulatorConfig(dt_ms=1.0, input_current=0.0, synaptic_scale=20.0))

    print("Initialized network:")
    for n in net.neurons:
        print(f"  neuron {n.id}: v={n.v:.1f}, u={n.u:.1f}, a={n.a}, b={n.b}, c={n.c}, d={n.d}")

    # Kick neuron 0 to start activity
    sim.inject_current_once(neuron_id=0, current=15.0)

    print("\nSimulating (spikes shown as *):")
    steps = 150
    for t in range(steps):
        sim.step()
        spikes_symbols = ["*" if n.spiked else "." for n in net.neurons]
        voltages = [round(n.v, 1) for n in net.neurons]
        print(f"t={t:04d}  spikes={' '.join(spikes_symbols)}  v={voltages}")


if __name__ == "__main__":
    main()


