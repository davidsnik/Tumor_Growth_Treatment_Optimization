from __future__ import annotations
from pathlib import Path

import pyvista as pv
import time

def get_max(reader, times: list[float]) -> float:
    peak = 0.0
    for t in times:
        reader.set_active_time_value(t)
        grid = reader.read()
        if isinstance(grid, pv.MultiBlock):
            grid = grid.combine()
        for pop in ("sensitive", "resistant"):
            if pop in grid.array_names:
                peak = max(peak, float(grid[pop].max()))
    return peak

def plot_tumor_interactive(tumor_data_path: Path) -> None:
    if not tumor_data_path.exists():
        raise FileNotFoundError(f"{tumor_data_path} does not exits")
    if tumor_data_path.suffix not in (".vtu", ".pvd"):
        raise ValueError(f"Function expects a .vtu or .pvd file not, {tumor_data_path.suffix}")

    reader = pv.get_reader(str(tumor_data_path))
    times = reader.time_values

    max_tumor_concentration = get_max(reader, times)
    if max_tumor_concentration == 0.0:
        raise ValueError(f"The max tumor concentration in the data is 0.0, check the data or simulations.")

    ### Load the data beforehand
    frames = []
    for t in times:
        reader.set_active_time_value(t)
        grid = reader.read()
        if isinstance(grid, pv.MultiBlock):
            grid = grid.combine()
        frames.append(grid)

    state = {"time_index": 0, "pop": "sensitive", "threshold": 0.0, "playing": False}
    plotter = pv.Plotter()
    actor = None

    def render_new():
        nonlocal actor

        grid = frames[state["time_index"]]
        pop = state["pop"]
        thresh = state["threshold"]
        shown = grid.threshold(thresh, scalars=pop) if thresh > 0.0 else grid

        if actor is not None:
            plotter.remove_actor(actor, render=False)
            actor = None

        if shown.n_points == 0:
            plotter.render()
            return

        actor = plotter.add_mesh(
            shown,
            scalars=pop,
            cmap="viridis" if pop == "sensitive" else "magma",
            clim=[0.0, max_tumor_concentration],
            show_scalar_bar=True,
        )
        plotter.render()

    ### Possible actions a user can take

    def set_time(value):
        state["time_index"] = int(round(value))
        render_new()

    def set_threshold(value):
        state["threshold"] = value
        render_new()

    def toggle_pop(res):
        state["pop"] = "resistant" if res else "sensitive"
        render_new()

    def toggle_play(flag):
        state["playing"] = flag
        if not flag:
            return

        if state["time_index"] >= len(times) - 1:
            state["time_index"] = 0
            time_slider.GetRepresentation().SetValue(0)
            render_new()

        while state["playing"]:
            if state["time_index"] < len(times) - 1:
                state["time_index"] += 1
                time_slider.GetRepresentation().SetValue(state["time_index"])
                render_new()
            else:
                state["playing"] = False
                play_button.GetRepresentation().SetState(0)
                render_new()
                break
            
            plotter.iren.process_events()
            time.sleep(0.01)

    ### Define all the sliders and buttons

    time_slider = plotter.add_slider_widget(
        set_time,
        [0, len(times) - 1],
        title=f"Time (t={times[0]:.2f}..{times[-1]:.2f})",
        pointa=(0.05, 0.9),
        pointb=(0.45, 0.9),
    )
    
    plotter.add_slider_widget(
        set_threshold,
        [0.0, max_tumor_concentration],
        title="Min density shown",
        pointa=(0.55, 0.9),
        pointb=(0.95, 0.9),
    )
    
    plotter.add_checkbox_button_widget(toggle_pop, value=False, position=(10, 10))
    plotter.add_text("Checkbox: unchecked=sensitive, checked=resistant", position="lower_right", font_size=9)

    play_button = plotter.add_checkbox_button_widget(
        toggle_play, 
        value=False, 
        position=(10, 60)
    )

    render_new()
    plotter.show()


if __name__=="__main__":
    plot_tumor_interactive(Path("outputs/raw/tumor_data.pvd"))