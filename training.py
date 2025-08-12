from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
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
    reset_all_states(model)
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


def simple_delta_train_step(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta: float,
    *,
    random_sign: bool = True,
) -> Tuple[float, float, bool]:
    """Run one simple delta training step.

    Returns (baseline_error, candidate_error, accepted)
    and mutates model weights if accepted.
    """
    base_weights = _get_weight_vector(model)
    baseline_pred = run_cycle_and_read_output(model, sim_config, steps)
    baseline_err = squared_error(target_output, baseline_pred)

    # Propose a candidate by adding +/- delta to each weight
    signs = [random.choice([-1.0, 1.0]) if random_sign else 1.0 for _ in base_weights]
    candidate = [w + s * delta for w, s in zip(base_weights, signs)]
    _set_weight_vector(model, candidate)

    candidate_pred = run_cycle_and_read_output(model, sim_config, steps)
    candidate_err = squared_error(target_output, candidate_pred)

    if candidate_err <= baseline_err:
        # keep
        accepted = True
    else:
        # revert
        _set_weight_vector(model, base_weights)
        accepted = False

    return baseline_err, candidate_err, accepted


def simple_delta_train(
    model: Network,
    sim_config: SimulatorConfig,
    target_output: List[bool],
    steps: int,
    delta: float,
    iterations: int = 10,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, bool]]:
    """Iteratively apply simple delta training.

    Returns a history list of (baseline_error, candidate_error, accepted).
    """
    if seed is not None:
        random.seed(seed)
    history: List[Tuple[float, float, bool]] = []
    for _ in range(iterations):
        baseline_err, candidate_err, accepted = simple_delta_train_step(
            model, sim_config, target_output, steps, delta
        )
        history.append((baseline_err, candidate_err, accepted))
    return history


__all__ = [
    "run_cycle_and_read_output",
    "squared_error",
    "simple_delta_train_step",
    "simple_delta_train",
]


