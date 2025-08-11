from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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
    spiked: bool = False

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


