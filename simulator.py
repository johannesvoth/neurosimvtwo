from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from network import Network


@dataclass
class SimulatorConfig:
    dt_ms: float = 1.0
    input_current: float = 0.0
    synaptic_scale: float = 1.0


@dataclass
class Simulator:
    model: Network
    config: SimulatorConfig = field(default_factory=SimulatorConfig)

    # one-step input injection per neuron id
    _pending_injection: Dict[int, float] = field(default_factory=dict, init=False)

    def inject_current_once(self, neuron_id: int, current: float) -> None:
        self._pending_injection[neuron_id] = self._pending_injection.get(neuron_id, 0.0) + current

    def step(self) -> None:
        self._step_euler()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def _step_euler(self) -> None:
        dt = self.config.dt_ms

        # Compute synaptic input for each neuron from spikes in previous step
        incoming: Dict[int, float] = {n.id: 0.0 for n in self.model.neurons}
        for c in self.model.connections:
            src = self.model.find_neuron(c.source_id)
            if src and src.spiked:
                incoming[c.target_id] = incoming.get(c.target_id, 0.0) + c.weight * self.config.synaptic_scale

        # Update all neurons using Euler method
        for n in self.model.neurons:
            # I is global baseline + synaptic + any one-step injection queued
            I = (
                self.config.input_current
                + incoming.get(n.id, 0.0)
                + self._pending_injection.get(n.id, 0.0)
            )

            # Izhikevich model differential equations (ms scale)
            # dv/dt = 0.04 v^2 + 5 v + 140 - u + I
            # du/dt = a (b v - u)
            dv = 0.04 * n.v * n.v + 5.0 * n.v + 140.0 - n.u + I
            du = n.a * (n.b * n.v - n.u)

            # Euler update
            n.v += dt * dv
            n.u += dt * du

            # Spike condition
            if n.v >= 30.0:
                n.v = n.c
                n.u += n.d
                n.spiked = True
            else:
                n.spiked = False

        # clear one-shot injections
        self._pending_injection.clear()


__all__ = ["Simulator", "SimulatorConfig"]


