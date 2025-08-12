from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import random


@dataclass
class Neuron:
    """Izhikevich-style neuron for use in a simple network simulator.

    This struct stores state and parameters used by the Euler update.
    """

    id: int
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v: float = -65.0
    u: float = -13.0
    i_const: float = 0.0
    spiked: bool = False

    def set_constant_current(self, current: float) -> None:
        """Set a constant input current specific to this neuron.

        This value is added to synaptic and global inputs during simulation.
        """
        self.i_const = float(current)

    # Convenience presets similar to Izhikevich (2003)
    @classmethod
    def regular_spiking(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.02, b=0.2, c=-65.0, d=8.0)

    @classmethod
    def intrinsically_bursting(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.02, b=0.2, c=-55.0, d=4.0)

    @classmethod
    def chattering(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.02, b=0.2, c=-50.0, d=2.0)

    @classmethod
    def fast_spiking(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.10, b=0.2, c=-65.0, d=2.0)

    @classmethod
    def low_threshold_spiking(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.02, b=0.25, c=-65.0, d=2.0)

    @classmethod
    def resonator(cls, neuron_id: int) -> "Neuron":
        return cls(id=neuron_id, a=0.10, b=0.26, c=-65.0, d=2.0)


@dataclass
class Connection:
    source_id: int
    target_id: int
    weight: float = 1.0


@dataclass
class Network:
    neurons: List[Neuron] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)

    def find_neuron(self, neuron_id: int) -> Optional[Neuron]:
        for neuron in self.neurons:
            if neuron.id == neuron_id:
                return neuron
        return None


__all__ = ["Neuron", "Connection", "Network"]


# Pixel input neuron encodes a single binary pixel: white -> constant current, black -> 0
@dataclass
class PixelInputNeuron(Neuron):
    # Lower default d (reset increment) for pixel-driven inputs
    d: float = .2
    i_on: float = 20.0

    def encode(self, is_white: bool) -> None:
        self.set_constant_current(self.i_on if is_white else 0.0)

    @classmethod
    def create(
        cls, neuron_id: int, i_on: float = 20.0, d: Optional[float] = None
    ) -> "PixelInputNeuron":
        return cls(id=neuron_id, i_on=i_on, d=(cls.d if d is None else d))


def make_pixel_input_neurons(
    start_id: int, count: int, i_on: float = 20.0, d: Optional[float] = None
) -> List[PixelInputNeuron]:
    return [
        PixelInputNeuron.create(neuron_id=start_id + idx, i_on=i_on, d=d)
        for idx in range(count)
    ]


def show_random_binary_image(inputs: List[PixelInputNeuron], white_probability: float = 0.5) -> List[bool]:
    """Randomly assign each input neuron to white (True) or black (False).

    Returns the list of assigned booleans for convenience.
    """
    assignments: List[bool] = []
    for n in inputs:
        is_white = random.random() < white_probability
        n.encode(is_white)
        assignments.append(is_white)
    return assignments


__all__ += ["PixelInputNeuron", "make_pixel_input_neurons", "show_random_binary_image"]


# Hidden-layer utilities
def make_hidden_neurons(
    start_id: int,
    count: int,
    preset: str = "fast_spiking",
    *,
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
) -> List[Neuron]:
    """Create a list of hidden neurons using a named preset or custom overrides.

    Supported presets are the classmethods on Neuron: regular_spiking, intrinsically_bursting,
    chattering, fast_spiking, low_threshold_spiking, resonator.
    If any of a/b/c/d are provided, they will override the chosen preset per neuron.
    """
    if not hasattr(Neuron, preset):
        raise ValueError(f"Unknown preset '{preset}'.")

    factory = getattr(Neuron, preset)
    neurons: List[Neuron] = []
    for idx in range(count):
        n = factory(neuron_id=start_id + idx)
        if a is not None:
            n.a = a
        if b is not None:
            n.b = b
        if c is not None:
            n.c = c
        if d is not None:
            n.d = d
        neurons.append(n)
    return neurons


def fully_connect(
    neuron_ids: List[int], weight: float = 1.0, include_self: bool = False
) -> List[Connection]:
    """Create all-to-all connections between the given neuron IDs.

    By default, self-connections are excluded.
    """
    connections: List[Connection] = []
    for src in neuron_ids:
        for tgt in neuron_ids:
            if not include_self and src == tgt:
                continue
            connections.append(Connection(source_id=src, target_id=tgt, weight=weight))
    return connections


__all__ += ["make_hidden_neurons", "fully_connect"]


def reset_all_states(model: Network) -> None:
    """Reset neuron states for a fair evaluation start.

    - Membrane potential v -> c
    - Recovery variable u -> b*v
    - Clear spike flags
    - Clear output latches
    """
    for n in model.neurons:
        n.v = n.c
        n.u = n.b * n.v
        n.spiked = False
        # Clear any output latch state if present
        if isinstance(n, PixelOutputNeuron):  # type: ignore[name-defined]
            n.on_steps_remaining = 0


# Pixel output neuron decodes a single binary pixel from activity
@dataclass
class PixelOutputNeuron(Neuron):
    # Number of steps to stay "on" after a spike
    on_duration_steps: int = 10
    # Internal countdown of remaining "on" steps
    on_steps_remaining: int = 0

    def decode(self) -> bool:
        return self.on_steps_remaining > 0

    @classmethod
    def create(
        cls, neuron_id: int, on_duration_steps: int = 5
    ) -> "PixelOutputNeuron":
        return cls(id=neuron_id, on_duration_steps=on_duration_steps)


def make_pixel_output_neurons(
    start_id: int, count: int, on_duration_steps: int = 5
) -> List[PixelOutputNeuron]:
    return [
        PixelOutputNeuron.create(neuron_id=start_id + idx, on_duration_steps=on_duration_steps)
        for idx in range(count)
    ]


def read_output_binary_image(outputs: List[PixelOutputNeuron]) -> List[bool]:
    return [n.decode() for n in outputs]


__all__ += [
    "PixelOutputNeuron",
    "make_pixel_output_neurons",
    "read_output_binary_image",
]


