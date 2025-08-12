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
    run_cycle_and_average_score,
)
from pathlib import Path
import json
import shutil
from visualize import visualize_latest


def build_demo_network(num_inputs: int = 8) -> Network:
    # Tunables for hidden layer
    hidden_count = 8 # number of hidden neurons
    input_to_hidden_weight = .5
    hidden_recurrent_weight = 0.5
    hidden_to_output_weight = .5

    input_neurons: List[PixelInputNeuron] = make_pixel_input_neurons(
        start_id=0, count=num_inputs, i_on=20.0, d=0.2
    )

    hidden_neurons: List[Neuron] = make_hidden_neurons(
        start_id=num_inputs, count=hidden_count, preset="fast_spiking"
    )

    # Output neurons mirror the input size so it forms an image
    output_neurons: List[PixelOutputNeuron] = make_pixel_output_neurons(
        start_id=num_inputs + hidden_count, count=num_inputs, on_duration_steps= 15
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

    # Manually coded single image for input neurons (True=white/ON, False=black/OFF)
    manual_pattern: List[bool] = [True, False, True, False, True, False, True, False]
    inputs = [n for n in net.neurons if isinstance(n, PixelInputNeuron)]
    for n, is_white in zip(inputs, manual_pattern):
        n.encode(is_white) # set the input current to the value of the pixel
    print("Input image:", manual_pattern)

    # Training routine: per-step average scoring on latched outputs
    train_steps = 50
    average_mode = "infinite"  # "fixed" or "infinite"
    window_size = 50
    delta = 0.2
    epochs = 3000
    perturbation_ratio = 1

    # Initial score using average over steps
    #initial_score = run_cycle_and_average_score(
    #    net, sim.config, manual_pattern, steps=train_steps, average_mode=average_mode, window_size=window_size
    #)
    #print("Initial average score:", round(initial_score, 3))

    results_dir = Path("runs")
    # Clear previous runs for a fresh session
    if results_dir.exists():
        try:
            for p in results_dir.iterdir():
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
        except Exception as e:
            print("Warning: failed to fully clear runs directory:", e)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Saving epoch snapshots to:", str(results_dir.resolve()))

    for idx in range(1, epochs + 1):
        baseline_score, candidate_score, accepted = simple_delta_train_step(
            net,
            sim.config,
            manual_pattern,
            steps=train_steps,
            delta=delta,
            average_mode=average_mode,
            window_size=window_size,
            perturbation_ratio=perturbation_ratio,
        )
        print(
            f"Epoch {idx:04d}: baseline={baseline_score:.3f} candidate={candidate_score:.3f} accepted={accepted}",
            flush=True,
        )

        # Persist snapshot for the visualizer to load (every 10 epochs)
        if idx % 10 == 0:
            snapshot = {
                "epoch": idx,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "accepted": accepted,
                "weights": [c.weight for c in net.connections],
            }
            (results_dir / f"epoch_{idx:05d}.json").write_text(json.dumps(snapshot, indent=2))

    # Reset state and re-encode inputs for a clean rollout after training
    reset_all_states(net)
    for n, is_white in zip(inputs, manual_pattern):
        n.encode(is_white)

    # Final evaluation using the same average scoring
    #final_score = run_cycle_and_average_score(
    #    net, sim.config, manual_pattern, steps=train_steps, average_mode=average_mode, window_size=window_size
    #)
    #print("Final average score:", round(final_score, 3))

    # Optional visualization (runs after training with current weights)
    try:
        # Pre-roll to the average-evaluation horizon, then pause
        visualize_latest(net,sim.config, steps_per_frame=1,eval_steps=train_steps,preroll=False,)

        from visualize import visualize_graph_latest
        visualize_graph_latest(net, sim.config, steps_per_frame=1, preroll=False)
    
    except Exception as e:
        print("Visualization failed:", e)


if __name__ == "__main__":
    main()


