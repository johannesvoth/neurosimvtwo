from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
from pathlib import Path
import json
import random

from network import Network, PixelOutputNeuron
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


