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
    prune_small_weights_train_step,
    i_const_delta_train_step,
    meta_tune_hyperparams,
)
from pathlib import Path
import json
import shutil
from visualize import visualize_latest
import random


def build_demo_network(num_inputs: int = 8) -> Network:
    # Tunables for hidden layer
    hidden_count = 16 # number of hidden neurons
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

    net = Network(neurons=neurons, connections=connections)
    # Initialize all connection weights uniformly in [-1, 1]
    for c in net.connections:
        c.weight = random.uniform(-1.0, 1.0)
    return net


def main() -> None:
    net = build_demo_network(num_inputs=8)
    sim = Simulator(model=net, config=SimulatorConfig(dt_ms=1.0, input_current=0.0))

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
    delta = 0.24
    i_delta = 5.0
    epochs = 3000
    perturbation_ratio = 1

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
        # Primary candidate: weight perturbations
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
        print(f"Epoch {idx:04d}: baseline={baseline_score:.3f} candidate={candidate_score:.3f} accepted={accepted}", flush=True)

        # Independent candidate: adjust neuron i_const (exclude inputs)
        ic_base, ic_cand, ic_accept = i_const_delta_train_step(
            net,
            sim.config,
            manual_pattern,
            steps=train_steps,
            delta_i=i_delta,
            average_mode=average_mode,
            window_size=window_size,
            perturbation_ratio=perturbation_ratio,
        )
        print(f"Epoch {idx:04d} IC: baseline={ic_base:.3f} candidate={ic_cand:.3f} accepted={ic_accept}", flush=True)

        # After 500 epochs, every 10th epoch run a separate pruning candidate step, independent of weight perturbations
        pruned_accepted = None
        prune_baseline = None
        prune_candidate = None
        if idx >= 500 and idx % 10 == 0:
            prune_baseline, prune_candidate, pruned_accepted = prune_small_weights_train_step(
                net,
                sim.config,
                manual_pattern,
                steps=train_steps,
                threshold=0.1,
                average_mode=average_mode,
                window_size=window_size,
            )
            print(
                f"Epoch {idx:04d} PRUNE: baseline={prune_baseline:.3f} candidate={prune_candidate:.3f} accepted={pruned_accepted}",
                flush=True,
            )

        # Every 100 epochs, propose meta-updates to delta and i_delta via short-horizon evaluation
        if idx % 100 == 0:
            new_delta, new_i_delta, delta_acc, i_delta_acc = meta_tune_hyperparams(
                net,
                sim.config,
                manual_pattern,
                current_delta=delta,
                current_i_delta=i_delta,
                train_steps=train_steps,
                average_mode=average_mode,
                window_size=window_size,
                meta_epochs=100,
                delta_rel_step=0.2,
                i_delta_rel_step=0.2,
                prune_threshold=0.1,
                prune_every=10,
                prune_start_epoch=500,
            )
            print(
                f"Epoch {idx:04d} META: delta {delta:.3f}->{new_delta:.3f} ({'acc' if delta_acc else 'rej'}), "
                f"i_delta {i_delta:.3f}->{new_i_delta:.3f} ({'acc' if i_delta_acc else 'rej'})",
                flush=True,
            )
            delta, i_delta = new_delta, new_i_delta

        # Persist snapshot for the visualizer to load (every 10 epochs)
        if idx % 10 == 0:
            snapshot = {
                "epoch": idx,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "accepted": accepted,
                "prune_baseline_score": prune_baseline,
                "prune_candidate_score": prune_candidate,
                "prune_accepted": pruned_accepted,
                "weights": [c.weight for c in net.connections],
                "i_consts": [n.i_const for n in net.neurons],
                "delta": delta,
                "i_delta": i_delta,
            }
            (results_dir / f"epoch_{idx:05d}.json").write_text(json.dumps(snapshot, indent=2))

    # Reset state and re-encode inputs for a clean rollout after training
    reset_all_states(net)
    for n, is_white in zip(inputs, manual_pattern):
        n.encode(is_white)

    try:
        # Pre-roll to the average-evaluation horizon, then pause
        visualize_latest(net,sim.config, steps_per_frame=1,eval_steps=train_steps,preroll=False,)

        from visualize import visualize_graph_latest
        visualize_graph_latest(net, sim.config, steps_per_frame=1, preroll=False)
    
    except Exception as e:
        print("Visualization failed:", e)


if __name__ == "__main__":
    main()


