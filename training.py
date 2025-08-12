from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
import copy
from pathlib import Path
import json
import random

from network import Network, PixelOutputNeuron, PixelInputNeuron
from simulator import Simulator, SimulatorConfig
from network import read_output_binary_image
from network import reset_all_states


def _get_weight_vector(model: Network) -> List[float]:
    return [c.weight for c in model.connections]


def _set_weight_vector(model: Network, weights: List[float]) -> None:
    if len(weights) != len(model.connections):
        raise ValueError("Weight vector length mismatch with model connections")
    for c, w in zip(model.connections, weights):
        c.weight = w


# -------------------------
# Simple delta training API
# -------------------------

def _booleans_to_ints(values: Sequence[bool]) -> List[int]:
    return [1 if v else 0 for v in values]


def run_cycle_and_read_output(
    model: Network, sim_config: SimulatorConfig, steps: int
) -> List[bool]:
    # reset_all_states(model)
    sim = Simulator(model=model, config=sim_config)
    for _ in range(steps):
        sim.step()
    outputs: List[PixelOutputNeuron] = [
        n for n in model.neurons if isinstance(n, PixelOutputNeuron)
    ]
    return read_output_binary_image(outputs)


def squared_error(target: Sequence[bool], prediction: Sequence[bool]) -> float:
    if len(target) != len(prediction):
        raise ValueError("target and prediction must have the same length")
    t = _booleans_to_ints(target)
    y = _booleans_to_ints(prediction)
    return float(sum((yi - ti) * (yi - ti) for yi, ti in zip(y, t)))


def run_cycle_and_average_score(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    *,
    steps: int,
    average_mode: str = "fixed",
    window_size: int = 100,
) -> float:
    """Run a cycle and return average per-step accuracy over a window.

    - Each step decodes via read_output_binary_image and computes fraction of pixels matching target.
    - If average_mode == 'infinite', averages over all 'steps'.
    - If average_mode == 'fixed', averages over the last 'window_size' steps.
    """
    reset_all_states(model)
    sim = Simulator(model=model, config=sim_config)
    outputs: List[PixelOutputNeuron] = [
        n for n in model.neurons if isinstance(n, PixelOutputNeuron)
    ]
    if not outputs or len(outputs) != len(target_output):
        return 0.0

    per_step_scores: List[float] = []
    denom = float(len(target_output)) # number of pixels in the image
    for _ in range(max(1, steps)):
        sim.step()
        pred = read_output_binary_image(outputs)
        matches = sum(1 for p, t in zip(pred, target_output) if p == t) # count the number of matches, the if p == t is true, then it is a match. counts for 0 and 1.
        per_step_scores.append(matches / denom) # matches divided by the number of pixels in the image, so the total amount. This is the accuracy of the model.

    if average_mode == "infinite":
        return sum(per_step_scores) / len(per_step_scores) # so it might be a big number since its a sum of all the scores. We simply average it by the length of the list to get the average score.

    k = min(window_size, len(per_step_scores))
    #If you've only run for 20 steps, but your window_size is 50, you can't possibly look at the last 50 scores. In this case, len(per_step_scores) is 20, so k becomes 20.
    return sum(per_step_scores[-k:]) / float(k)


## (Windowed objective removed from active training path; latch-based endpoint used instead)


def simple_delta_train_step(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta: float,
    *,
    average_mode: str = "fixed",
    window_size: int,
    perturbation_ratio: float = 1.0,
    random_sign: bool = True,
) -> Tuple[float, float, bool]:
    """Run one simple delta training step with windowed average scoring.

    Returns (baseline_score, candidate_score, accepted)
    and mutates model weights if accepted.
    """
    base_weights = _get_weight_vector(model)
    baseline_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    # Propose a candidate by perturbing a subset of weights
    num_weights = len(base_weights)
    ratio = max(0.0, min(1.0, perturbation_ratio))
    selected = [random.random() < ratio for _ in range(num_weights)]
    if ratio > 0.0 and not any(selected):
        # Ensure at least one change when ratio > 0
        selected[random.randrange(num_weights)] = True

    candidate = list(base_weights)
    for i, sel in enumerate(selected):
        if not sel:
            continue
        sign = random.choice([-1.0, 1.0]) if random_sign else 1.0
        candidate[i] = candidate[i] + sign * delta
    _set_weight_vector(model, candidate)

    candidate_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    if candidate_score > baseline_score + 1e-9:
        # keep
        accepted = True
    else:
        # revert
        _set_weight_vector(model, base_weights)
        accepted = False

    return baseline_score, candidate_score, accepted


def simple_delta_train(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta: float,
    iterations: int = 10,
    seed: Optional[int] = None,
    *,
    average_mode: str = "fixed",
    window_size: int = 100,
    perturbation_ratio: float = 1.0,
) -> List[Tuple[float, float, bool]]:
    """Iteratively apply simple delta training.

    Returns a history list of (baseline_error, candidate_error, accepted).
    """
    if seed is not None:
        random.seed(seed)
    history: List[Tuple[float, float, bool]] = []
    results_dir = Path("runs")
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch_idx in range(1, iterations + 1):
        baseline_score, candidate_score, accepted = simple_delta_train_step(
            model,
            sim_config,
            target_output,
            steps,
            delta,
            average_mode=average_mode,
            window_size=window_size,
            perturbation_ratio=perturbation_ratio,
        )
        history.append((baseline_score, candidate_score, accepted))

        # Persist epoch snapshot (weights and metrics)
        snapshot = {
            "epoch": epoch_idx,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "accepted": accepted,
            "weights": [c.weight for c in model.connections],
        }
        (results_dir / f"epoch_{epoch_idx:05d}.json").write_text(
            json.dumps(snapshot, indent=2)
        )
    return history


__all__ = [
    "run_cycle_and_read_output",
    "squared_error",
    "simple_delta_train_step",
    "simple_delta_train",
    "run_cycle_and_average_score",
    "train_average_without_epochs",
    "prune_small_weights_train_step",
    "i_const_delta_train_step",
    "clone_network",
    "run_training_epochs",
    "meta_tune_hyperparams",
]


def train_average_without_epochs(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta: float,
    *,
    average_mode: str = "fixed",
    window_size: int = 100,
    perturbation_ratio: float = 1.0,
    random_sign: bool = True,
) -> Tuple[float, float, bool]:
    """Run a single average-scored training step (no epochs, no persistence).

    Returns (baseline_score, candidate_score, accepted).
    """
    return simple_delta_train_step(
        model,
        sim_config,
        target_output,
        steps,
        delta,
        average_mode=average_mode,
        window_size=window_size,
        perturbation_ratio=perturbation_ratio,
        random_sign=random_sign,
    )


def prune_small_weights_train_step(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    *,
    steps: int,
    threshold: float = 0.1,
    average_mode: str = "fixed",
    window_size: int = 100,
) -> Tuple[float, float, bool]:
    """Propose a candidate by zeroing weights with small magnitude and accept if score improves.

    Returns (baseline_score, candidate_score, accepted) and mutates model weights if accepted.
    """
    base_weights = _get_weight_vector(model)
    baseline_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    # Create pruning candidate: zero-out weights with |w| < threshold
    candidate = [0.0 if abs(w) < threshold else w for w in base_weights]
    _set_weight_vector(model, candidate)

    candidate_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    if candidate_score > baseline_score + 1e-9:
        accepted = True
    else:
        _set_weight_vector(model, base_weights)
        accepted = False

    return baseline_score, candidate_score, accepted


def _get_i_const_vector(model: Network) -> List[float]:
    return [n.i_const for n in model.neurons]


def _set_i_const_vector(model: Network, i_consts: List[float]) -> None:
    if len(i_consts) != len(model.neurons):
        raise ValueError("i_const vector length mismatch with model neurons")
    for n, v in zip(model.neurons, i_consts):
        n.i_const = v


def i_const_delta_train_step(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta_i: float,
    *,
    average_mode: str = "fixed",
    window_size: int = 100,
    perturbation_ratio: float = 1.0,
    random_sign: bool = True,
) -> Tuple[float, float, bool]:
    """Propose a candidate by perturbing neuron i_const values (excluding inputs) and accept if score improves."""
    base_consts = _get_i_const_vector(model)
    baseline_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    # Select a subset of neurons to modify (exclude PixelInputNeuron to preserve the external input pattern)
    num_neurons = len(model.neurons)
    selected = [False] * num_neurons
    ratio = max(0.0, min(1.0, perturbation_ratio))
    for i, n in enumerate(model.neurons):
        if isinstance(n, PixelInputNeuron):
            continue
        selected[i] = (random.random() < ratio)
    if ratio > 0.0 and not any(selected):
        # ensure at least one eligible neuron is changed
        eligible_indices = [i for i, n in enumerate(model.neurons) if not isinstance(n, PixelInputNeuron)]
        if eligible_indices:
            selected[random.choice(eligible_indices)] = True

    candidate = list(base_consts)
    for i, sel in enumerate(selected):
        if not sel:
            continue
        sign = random.choice([-1.0, 1.0]) if random_sign else 1.0
        candidate[i] = candidate[i] + sign * delta_i
    _set_i_const_vector(model, candidate)

    candidate_score = run_cycle_and_average_score(
        model, sim_config, target_output, steps=steps, average_mode=average_mode, window_size=window_size
    )

    if candidate_score > baseline_score + 1e-9:
        accepted = True
    else:
        _set_i_const_vector(model, base_consts)
        accepted = False

    return baseline_score, candidate_score, accepted


def clone_network(model: Network) -> Network:
    """Deep copy the network so we can branch training safely."""
    return copy.deepcopy(model)


def run_training_epochs(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    *,
    epochs: int,
    train_steps: int,
    delta: float,
    i_delta: float,
    average_mode: str = "fixed",
    window_size: int = 100,
    perturbation_ratio: float = 1.0,
    prune_threshold: float = 0.1,
    prune_every: int = 10,
    prune_start_epoch: int = 500,
) -> float:
    """Run a compact training loop (no persistence) mirroring main.py behavior, return final score."""
    for e in range(1, epochs + 1):
        simple_delta_train_step(
            model,
            sim_config,
            target_output,
            steps=train_steps,
            delta=delta,
            average_mode=average_mode,
            window_size=window_size,
            perturbation_ratio=perturbation_ratio,
        )

        i_const_delta_train_step(
            model,
            sim_config,
            target_output,
            steps=train_steps,
            delta_i=i_delta,
            average_mode=average_mode,
            window_size=window_size,
            perturbation_ratio=perturbation_ratio,
        )

        if e >= prune_start_epoch and e % prune_every == 0:
            prune_small_weights_train_step(
                model,
                sim_config,
                target_output,
                steps=train_steps,
                threshold=prune_threshold,
                average_mode=average_mode,
                window_size=window_size,
            )

    final_score = run_cycle_and_average_score(
        model,
        sim_config,
        target_output,
        steps=train_steps,
        average_mode=average_mode,
        window_size=window_size,
    )
    return final_score


def meta_tune_hyperparams(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    *,
    current_delta: float,
    current_i_delta: float,
    train_steps: int,
    average_mode: str,
    window_size: int,
    meta_epochs: int = 100,
    delta_rel_step: float = 0.2,
    i_delta_rel_step: float = 0.2,
    prune_threshold: float = 0.1,
    prune_every: int = 10,
    prune_start_epoch: int = 500,
) -> Tuple[float, float, bool, bool]:
    """Try multiplicative perturbations of delta and i_delta and accept if 100-epoch outcome improves.

    Returns: (new_delta, new_i_delta, delta_accepted, i_delta_accepted).
    """
    # Evaluate baseline trajectory over meta_epochs
    baseline_model_for_delta = clone_network(model)
    baseline_score_for_delta = run_training_epochs(
        baseline_model_for_delta,
        sim_config,
        target_output,
        epochs=meta_epochs,
        train_steps=train_steps,
        delta=current_delta,
        i_delta=current_i_delta,
        average_mode=average_mode,
        window_size=window_size,
        prune_threshold=prune_threshold,
        prune_every=prune_every,
        prune_start_epoch=prune_start_epoch,
    )

    # Propose delta candidate
    delta_sign = random.choice([-1.0, 1.0])
    candidate_delta = max(1e-6, current_delta * (1.0 + delta_sign * delta_rel_step))
    candidate_model_for_delta = clone_network(model)
    candidate_score_for_delta = run_training_epochs(
        candidate_model_for_delta,
        sim_config,
        target_output,
        epochs=meta_epochs,
        train_steps=train_steps,
        delta=candidate_delta,
        i_delta=current_i_delta,
        average_mode=average_mode,
        window_size=window_size,
        prune_threshold=prune_threshold,
        prune_every=prune_every,
        prune_start_epoch=prune_start_epoch,
    )

    new_delta = candidate_delta if candidate_score_for_delta > baseline_score_for_delta + 1e-9 else current_delta
    delta_accepted = new_delta != current_delta

    # Evaluate baseline for i_delta (can reuse same baseline as above)
    baseline_score_for_i = baseline_score_for_delta
    # Propose i_delta candidate
    i_sign = random.choice([-1.0, 1.0])
    candidate_i_delta = max(1e-6, current_i_delta * (1.0 + i_sign * i_delta_rel_step))
    candidate_model_for_i = clone_network(model)
    candidate_score_for_i = run_training_epochs(
        candidate_model_for_i,
        sim_config,
        target_output,
        epochs=meta_epochs,
        train_steps=train_steps,
        delta=new_delta,  # use latest delta decision
        i_delta=candidate_i_delta,
        average_mode=average_mode,
        window_size=window_size,
        prune_threshold=prune_threshold,
        prune_every=prune_every,
        prune_start_epoch=prune_start_epoch,
    )

    new_i_delta = candidate_i_delta if candidate_score_for_i > baseline_score_for_i + 1e-9 else current_i_delta
    i_delta_accepted = new_i_delta != current_i_delta

    return new_delta, new_i_delta, delta_accepted, i_delta_accepted


