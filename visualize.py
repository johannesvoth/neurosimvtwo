from __future__ import annotations

from typing import List, Dict, Tuple
from pathlib import Path
import json

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore

from network import Network, PixelInputNeuron, PixelOutputNeuron, read_output_binary_image
from simulator import Simulator, SimulatorConfig


__all__ = ["visualize_latest", "visualize_graph_latest"]


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


def _layer_positions(
    model: Network,
    surface_size: Tuple[int, int],
) -> Dict[int, Tuple[int, int]]:
    width, height = surface_size
    margin_x = 120
    margin_y = 80

    inputs: List[PixelInputNeuron] = [n for n in model.neurons if isinstance(n, PixelInputNeuron)]
    outputs: List[PixelOutputNeuron] = [n for n in model.neurons if isinstance(n, PixelOutputNeuron)]
    hidden = [n for n in model.neurons if (n not in inputs and n not in outputs)]

    layers = [inputs, hidden, outputs]
    layer_x = [margin_x, width // 2, width - margin_x]

    positions: Dict[int, Tuple[int, int]] = {}
    for layer_idx, layer in enumerate(layers):
        if not layer:
            continue
        x = layer_x[layer_idx]
        gap = max(1, len(layer) - 1)
        for i, neuron in enumerate(layer):
            y = margin_y + int((height - 2 * margin_y) * (i / gap if gap > 0 else 0.5))
            positions[neuron.id] = (x, y)
    return positions


def _weight_color_and_thickness(weight: float, max_abs_weight: float) -> Tuple[Tuple[int, int, int], int]:
    max_abs_weight = max(max_abs_weight, 1e-6)
    intensity = min(1.0, abs(weight) / max_abs_weight)
    thickness = 1 + int(5 * intensity)
    if weight >= 0:
        # green scale
        g = int(150 + 105 * intensity)
        r = int(40 * (1 - intensity))
        b = int(40 * (1 - intensity))
        color = (r, g, b)
    else:
        # red scale
        r = int(150 + 105 * intensity)
        g = int(40 * (1 - intensity))
        b = int(40 * (1 - intensity))
        color = (r, g, b)
    return color, thickness


def _draw_quad_bezier(surface, color, p0, p1, p2, width: int = 2, segments: int = 20) -> None:
    # Approximate quadratic Bezier with line segments
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        points.append((int(x), int(y)))
    for i in range(len(points) - 1):
        pygame.draw.line(surface, color, points[i], points[i + 1], width)


def _color_for_neuron(n) -> Tuple[int, int, int]:
    if n.spiked:
        return (255, 220, 0)
    if isinstance(n, PixelInputNeuron):
        return (100, 180, 255)
    if isinstance(n, PixelOutputNeuron):
        return (210, 140, 255)
    return (200, 200, 200)


def visualize_graph_latest(
    model: Network,
    sim_config: SimulatorConfig,
    *,
    runs_dir: str = "runs",
    window_size: Tuple[int, int] = (1280, 860),
    fps: int = 30,
    steps_per_frame: int = 1,
    eval_steps: int = 0,
    preroll: bool = False,
) -> None:
    """Visualize the network graph with curved, color-coded weights and spiking neurons.

    - Loads the latest snapshot from runs_dir and applies weights
    - Curved arcs between neurons, colored green (positive) or red (negative) by magnitude
    - Neurons light up yellow when they spike
    - Adjustable steps_per_frame; SPACE pause; RIGHT single-step when paused; +/- adjust speed
    - Optional preroll for eval_steps before pausing
    """
    if pygame is None:
        print("pygame is not installed; visualization skipped. Install 'pygame' to enable.")
        return

    # Load latest snapshot
    snapshot_files = sorted(Path(runs_dir).glob("epoch_*.json"))
    latest = snapshot_files[-1] if snapshot_files else None
    if latest is None:
        print(f"No snapshots found in '{runs_dir}'. Run training first.")
        return

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

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Vibe NeuroSim v2 - Network Graph Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    simulator = Simulator(model=model, config=sim_config)
    running = True
    paused = not preroll
    advanced_steps = 0

    # For weight color normalization
    max_abs_weight = max((abs(c.weight) for c in model.connections), default=1.0)

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
                elif event.key == pygame.K_RIGHT and paused:
                    for _ in range(max(1, int(steps_per_frame))):
                        simulator.step()
                    advanced_steps += max(1, int(steps_per_frame))

        if not paused:
            step_chunk = max(1, int(steps_per_frame))
            for _ in range(step_chunk):
                simulator.step()
            advanced_steps += step_chunk
            if preroll and eval_steps > 0 and advanced_steps >= eval_steps:
                paused = True

        screen.fill((18, 22, 26))

        # Layout
        positions = _layer_positions(model, window_size)

        # Draw connections as curved arcs
        for c in model.connections:
            if c.source_id not in positions or c.target_id not in positions:
                continue
            p0 = positions[c.source_id]
            p2 = positions[c.target_id]
            # Control point for curvature: offset perpendicular to the line
            mx = (p0[0] + p2[0]) / 2
            my = (p0[1] + p2[1]) / 2
            dx = p2[0] - p0[0]
            dy = p2[1] - p0[1]
            # Perpendicular vector
            px = -dy
            py = dx
            # Normalize and scale curvature
            plen = max((px * px + py * py) ** 0.5, 1e-6)
            curvature = 0.15  # adjust for visual separation
            cx = mx + curvature * (px / plen) * 120
            cy = my + curvature * (py / plen) * 120

            color, thickness = _weight_color_and_thickness(c.weight, max_abs_weight)
            _draw_quad_bezier(screen, color, p0, (int(cx), int(cy)), p2, width=thickness, segments=24)

        # Draw neurons
        for n in model.neurons:
            if n.id not in positions:
                continue
            x, y = positions[n.id]
            color = _color_for_neuron(n)
            pygame.draw.circle(screen, color, (x, y), 14)
            # Outline
            pygame.draw.circle(screen, (30, 30, 30), (x, y), 16, 2)

        # Overlay status
        label = f"snapshot: {latest.name} | steps/frame: {int(steps_per_frame)} | steps: {advanced_steps} | fps: {fps}"
        if paused:
            label += "  [paused]"
        info = font.render(label, True, (220, 220, 220))
        screen.blit(info, (12, window_size[1] - 28))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


