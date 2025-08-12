from __future__ import annotations

from typing import List
from pathlib import Path
import json

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore

from network import Network, PixelInputNeuron, PixelOutputNeuron, read_output_binary_image
from simulator import Simulator, SimulatorConfig


__all__ = ["visualize_latest"]


def visualize_latest(
    model: Network,
    sim_config: SimulatorConfig,
    *,
    runs_dir: str = "runs",
    pixel_size: int = 80,
    padding: int = 22,
    fps: int = 30,
    steps_per_frame: int = 1,
    eval_steps: int = 120,
    preroll: bool = True,
) -> None:
    """Load latest snapshot from runs_dir and visualize as black/white pixels.

    - Applies weights from the highest epoch (latest) file in runs_dir
    - Top row: input pattern (True=white if PixelInputNeuron has i_const>0)
    - Bottom row: output pattern (latched readout)
    - Steps the simulator continuously so output can evolve
    - Visualization speed adjustable via steps_per_frame and +/- keys
    - If preroll=True, advance eval_steps automatically on start, then auto-pause to show the endpoint
    """
    if pygame is None:
        print("pygame is not installed; visualization skipped. Install 'pygame' to enable.")
        return

    # Load latest snapshot
    snapshot_files = sorted(Path(runs_dir).glob("epoch_*.json"))
    if not snapshot_files:
        print(f"No snapshots found in '{runs_dir}'. Run training first.")
        return
    latest = snapshot_files[-1]
    try:
        data = json.loads(latest.read_text())
        weights = data.get("weights", [])
        if len(weights) != len(model.connections):
            print("Snapshot connection mismatch; visualization aborted")
            return
        for c, w in zip(model.connections, weights):
            c.weight = float(w)
        # Reset state after applying weights
        for n in model.neurons:
            n.v = n.c
            n.u = n.b * n.v
            n.spiked = False
            if isinstance(n, PixelOutputNeuron):
                n.on_steps_remaining = 0
    except Exception as e:
        print("Failed to load latest snapshot:", e)
        return

    inputs: List[PixelInputNeuron] = [n for n in model.neurons if isinstance(n, PixelInputNeuron)]
    outputs: List[PixelOutputNeuron] = [n for n in model.neurons if isinstance(n, PixelOutputNeuron)]
    num_cols = max(len(inputs), len(outputs))
    status_bar_height = 64
    width = padding * 2 + num_cols * pixel_size + 100
    height = padding * 3 + 2 * pixel_size + status_bar_height  # two rows + status bar

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Vibe NeuroSim v2 - Latest Snapshot Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    simulator = Simulator(model=model, config=sim_config)
    running = True
    # If preroll is requested, start unpaused to run the pre-steps, then auto-pause
    paused = not preroll
    advanced_steps = 0

    def draw_row(values: List[bool], row_idx: int) -> None:
        y0 = padding + row_idx * (pixel_size + padding)
        for col, val in enumerate(values):
            x0 = padding + col * pixel_size
            color = (255, 255, 255) if val else (0, 0, 0)
            pygame.draw.rect(screen, color, (x0, y0, pixel_size, pixel_size))
            pygame.draw.rect(screen, (80, 80, 80), (x0, y0, pixel_size, pixel_size), 1)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    steps_per_frame = min(1000, steps_per_frame + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    # Manual single advance when paused
                    if paused:
                        step_chunk = max(1, int(steps_per_frame))
                        for _ in range(step_chunk):
                            simulator.step()
                        advanced_steps += step_chunk

        if not paused:
            # Advance multiple steps per frame for speed control
            step_chunk = max(1, int(steps_per_frame))
            for _ in range(step_chunk):
                simulator.step()
            advanced_steps += step_chunk
            # If we were in preroll mode and reached eval_steps, auto-pause
            if preroll and advanced_steps >= eval_steps:
                paused = True

        # Input pattern from constant currents
        input_values = [(n.i_const or 0.0) > 0.0 for n in inputs]
        # Output pattern from latched readout
        output_values = read_output_binary_image(outputs)

        screen.fill((22, 26, 30))
        draw_row(input_values, row_idx=0)
        draw_row(output_values, row_idx=1)
        # Overlay snapshot and speed info
        status = []
        status.append(f"snapshot: {latest.name}")
        status.append(f"steps/frame: {int(steps_per_frame)}")
        status.append(f"steps: {advanced_steps}")
        status.append(f"fps: {fps}")
        if preroll:
            done = min(advanced_steps, eval_steps)
            status.append(f"pre-roll: {done}/{eval_steps}")
        label = " | ".join(status)
        if paused:
            label += "  [paused]"
        info = font.render(label, True, (220, 220, 220))
        text_y = height - padding - font.get_height()
        screen.blit(info, (10, text_y))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

__all__ = ["visualize_latest"]


