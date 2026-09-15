#!/usr/bin/env python3
"""Interactive field-line visualizer for a brief acceleration event."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class SimulationParams:
    t0: float
    t: float
    t_max: float
    v: float
    dv: float
    delta_t: float
    t_acc: float
    x0: float
    c: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize field lines and photosphere growth for a brief acceleration."
    )
    parser.add_argument("--t0", type=float, default=0.0, help="Initial reference time t0.")
    parser.add_argument("--t", type=float, default=7.0, help="Current observation time.")
    parser.add_argument("--t-max", type=float, default=15.0, help="Maximum t slider value.")
    parser.add_argument("--v", type=float, default=0.35, help="Initial velocity v (units of c).")
    parser.add_argument("--delta-v", type=float, default=0.30, help="Velocity change delta v.")
    parser.add_argument("--delta-t", type=float, default=1.20, help="Acceleration duration delta t.")
    parser.add_argument("--t-acc", type=float, default=2.5, help="Acceleration start time t_acc.")
    parser.add_argument("--x0", type=float, default=0.0, help="Position x(t0)=x0.")
    parser.add_argument("--c", type=float, default=1.0, help="Speed of light c (simulation units).")
    parser.add_argument("--output", type=str, default=None, help="Optional output snapshot path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --output).",
    )
    return parser.parse_args()


def clamp_speed(v: float, c: float, margin: float = 1e-3) -> float:
    return float(np.clip(v, -(1.0 - margin) * c, (1.0 - margin) * c))


def apply_constraints(p: SimulationParams) -> SimulationParams:
    c = max(abs(p.c), 1e-6)
    t_max = max(p.t_max, p.t0 + 0.1)
    t = float(np.clip(p.t, p.t0, t_max))
    t_acc = max(p.t_acc, p.t0)
    delta_t = max(p.delta_t, 1e-3)

    v = clamp_speed(p.v, c)
    v2 = clamp_speed(v + p.dv, c)
    dv = v2 - v

    return SimulationParams(
        t0=p.t0,
        t=t,
        t_max=t_max,
        v=v,
        dv=dv,
        delta_t=delta_t,
        t_acc=t_acc,
        x0=p.x0,
        c=c,
    )


def state_at_time(p: SimulationParams) -> dict[str, float]:
    t_end = p.t_acc + p.delta_t
    x_acc = p.x0 + p.v * (p.t_acc - p.t0)

    accel = p.dv / max(p.delta_t, EPS)
    x_end = x_acc + p.v * p.delta_t + 0.5 * accel * p.delta_t * p.delta_t

    if p.t <= p.t_acc:
        tau = p.t - p.t0
        x_now = p.x0 + p.v * tau
        v_now = p.v
    elif p.t <= t_end:
        tau = p.t - p.t_acc
        x_now = x_acc + p.v * tau + 0.5 * accel * tau * tau
        v_now = p.v + accel * tau
    else:
        tau = p.t - t_end
        x_now = x_end + (p.v + p.dv) * tau
        v_now = p.v + p.dv

    x_old_now = p.x0 + p.v * (p.t - p.t0)
    r_outer = max(p.c * (p.t - p.t_acc), 0.0)
    r_inner = max(p.c * (p.t - t_end), 0.0)

    return {
        "t_end": t_end,
        "x_acc": x_acc,
        "x_end": x_end,
        "x_now": x_now,
        "x_old_now": x_old_now,
        "v_now": v_now,
        "r_outer": r_outer,
        "r_inner": r_inner,
    }


def uniform_velocity_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_charge: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    beta2 = min(beta * beta, 0.999999)
    pref = 1.0 - beta2

    dx = x_grid - x_charge
    dy = y_grid

    denom = np.power(np.maximum(dx * dx + pref * dy * dy, 1e-6), 1.5)
    ex = pref * dx / denom
    ey = pref * dy / denom

    core_mask = (dx * dx + dy * dy) < 0.02
    ex = np.where(core_mask, 0.0, ex)
    ey = np.where(core_mask, 0.0, ey)
    return ex, ey


def transition_weight(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    p: SimulationParams,
    st: dict[str, float],
) -> np.ndarray:
    w = np.zeros_like(x_grid)
    if p.t <= p.t_acc:
        return w

    d_outer = np.hypot(x_grid - st["x_acc"], y_grid) - st["r_outer"]
    inside_outer = d_outer <= 0.0

    if p.t < st["t_end"]:
        progress = np.clip((p.t - p.t_acc) / max(p.delta_t, EPS), 0.0, 1.0)
        w[inside_outer] = progress
        return w

    d_inner = np.hypot(x_grid - st["x_end"], y_grid) - st["r_inner"]
    inside_inner = d_inner <= 0.0
    outside_outer = d_outer > 0.0
    shell = (~inside_inner) & (~outside_outer)

    w[inside_inner] = 1.0
    if np.any(shell):
        dist_from_outer = np.maximum(-d_outer[shell], 0.0)
        dist_to_inner = np.maximum(d_inner[shell], 0.0)
        w[shell] = dist_from_outer / (dist_from_outer + dist_to_inner + EPS)

    return np.clip(w, 0.0, 1.0)


def draw_scene(
    ax: plt.Axes,
    status_text,
    p: SimulationParams,
) -> None:
    st = state_at_time(p)

    span = max(
        5.0,
        st["r_outer"] + 2.0,
        st["r_inner"] + 2.0,
        abs(st["x_now"] - st["x_old_now"]) + 3.0,
    )
    x_min = min(st["x_acc"], st["x_end"], st["x_now"], st["x_old_now"]) - span
    x_max = max(st["x_acc"], st["x_end"], st["x_now"], st["x_old_now"]) + span
    y_lim = max(3.5, 0.75 * span)

    x = np.linspace(x_min, x_max, 220)
    y = np.linspace(-y_lim, y_lim, 180)
    x_grid, y_grid = np.meshgrid(x, y)

    ex_old, ey_old = uniform_velocity_field(x_grid, y_grid, st["x_old_now"], p.v / p.c)
    ex_new, ey_new = uniform_velocity_field(
        x_grid,
        y_grid,
        st["x_now"],
        st["v_now"] / p.c,
    )
    w = transition_weight(x_grid, y_grid, p, st)

    ex = (1.0 - w) * ex_old + w * ex_new
    ey = (1.0 - w) * ey_old + w * ey_new

    strength = np.hypot(ex, ey)
    max_strength = max(np.max(strength), EPS)
    linewidth = 0.55 + 1.25 * (strength / max_strength)

    ax.clear()
    ax.set_aspect("equal")
    ax.streamplot(
        x,
        y,
        ex,
        ey,
        color="black",
        density=1.2,
        linewidth=linewidth,
        arrowsize=0.8,
        minlength=0.1,
    )

    if st["r_outer"] > 0.0:
        ax.add_patch(
            Circle(
                (st["x_acc"], 0.0),
                st["r_outer"],
                fill=False,
                lw=1.4,
                ls="--",
                color="0.35",
            )
        )
    if p.t >= st["t_end"] and st["r_inner"] > 0.0:
        ax.add_patch(
            Circle((st["x_end"], 0.0), st["r_inner"], fill=False, lw=2.2, color="black")
        )

    ax.plot(st["x_now"], 0.0, "ko", ms=6)
    ax.plot(st["x_old_now"], 0.0, "o", color="0.55", ms=5)
    ax.plot(st["x_acc"], 0.0, "ks", ms=4)
    ax.plot(st["x_end"], 0.0, "k^", ms=4)
    ax.axhline(0.0, color="0.75", lw=0.9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title("Field Lines with Finite-Time Acceleration and Expanding Photosphere")
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y")

    status_text.set_text(
        "Constraints: |v|<c, |v+dv|<c, t_acc>=t0 | "
        f"v={p.v:.3f}c, v+dv={p.v + p.dv:.3f}c, "
        f"r_outer=c(t-t_acc)={st['r_outer']:.3f}, "
        f"r_inner=c(t-(t_acc+dt))={st['r_inner']:.3f}"
    )


def build_ui(initial: SimulationParams) -> None:
    fig = plt.figure(figsize=(12.4, 9.2))
    ax_main = fig.add_axes([0.07, 0.42, 0.90, 0.54])
    ax_info = fig.add_axes([0.07, 0.36, 0.90, 0.03])
    ax_info.axis("off")
    status_text = ax_info.text(0.0, 0.5, "", fontsize=10, va="center")

    ax_t = fig.add_axes([0.12, 0.29, 0.70, 0.022])
    ax_v = fig.add_axes([0.12, 0.245, 0.70, 0.022])
    ax_dv = fig.add_axes([0.12, 0.20, 0.70, 0.022])
    ax_dt = fig.add_axes([0.12, 0.155, 0.70, 0.022])
    ax_tacc = fig.add_axes([0.12, 0.11, 0.70, 0.022])
    ax_reset = fig.add_axes([0.85, 0.105, 0.11, 0.075])

    t_slider = Slider(
        ax_t,
        "t",
        initial.t0,
        initial.t_max,
        valinit=initial.t,
        valstep=0.01,
        valfmt="%.2f",
    )
    v_slider = Slider(
        ax_v,
        "v",
        -0.99 * initial.c,
        0.99 * initial.c,
        valinit=initial.v,
        valstep=0.001,
        valfmt="%.3f",
    )
    dv_slider = Slider(
        ax_dv,
        "dv",
        -1.30 * initial.c,
        1.30 * initial.c,
        valinit=initial.dv,
        valstep=0.001,
        valfmt="%.3f",
    )
    dt_slider = Slider(
        ax_dt,
        "dt",
        0.001,
        8.0,
        valinit=initial.delta_t,
        valstep=0.001,
        valfmt="%.3f",
    )
    tacc_slider = Slider(
        ax_tacc,
        "t_acc",
        initial.t0,
        initial.t_max,
        valinit=max(initial.t0, initial.t_acc),
        valstep=0.01,
        valfmt="%.2f",
    )
    reset_btn = Button(ax_reset, "Reset")

    t_acc_marker = ax_t.axvline(initial.t_acc, color="tab:red", lw=1.2, alpha=0.9)
    t_end_marker = ax_t.axvline(initial.t_acc + initial.delta_t, color="tab:orange", lw=1.2, alpha=0.9)

    busy = {"value": False}

    def current_params() -> SimulationParams:
        return SimulationParams(
            t0=initial.t0,
            t=t_slider.val,
            t_max=initial.t_max,
            v=v_slider.val,
            dv=dv_slider.val,
            delta_t=dt_slider.val,
            t_acc=tacc_slider.val,
            x0=initial.x0,
            c=initial.c,
        )

    def sync_and_draw(_=None) -> None:
        if busy["value"]:
            return
        busy["value"] = True
        try:
            p = apply_constraints(current_params())

            if abs(v_slider.val - p.v) > 1e-12:
                v_slider.set_val(p.v)
            if abs(dv_slider.val - p.dv) > 1e-12:
                dv_slider.set_val(p.dv)
            if abs(dt_slider.val - p.delta_t) > 1e-12:
                dt_slider.set_val(p.delta_t)
            if abs(tacc_slider.val - p.t_acc) > 1e-12:
                tacc_slider.set_val(p.t_acc)
            if abs(t_slider.val - p.t) > 1e-12:
                t_slider.set_val(p.t)

            t_acc_marker.set_xdata([p.t_acc, p.t_acc])
            t_end = p.t_acc + p.delta_t
            t_end_marker.set_xdata([t_end, t_end])

            draw_scene(ax_main, status_text, p)
            fig.canvas.draw_idle()
        finally:
            busy["value"] = False

    def on_reset(_event) -> None:
        t_slider.reset()
        v_slider.reset()
        dv_slider.reset()
        dt_slider.reset()
        tacc_slider.reset()

    t_slider.on_changed(sync_and_draw)
    v_slider.on_changed(sync_and_draw)
    dv_slider.on_changed(sync_and_draw)
    dt_slider.on_changed(sync_and_draw)
    tacc_slider.on_changed(sync_and_draw)
    reset_btn.on_clicked(on_reset)

    sync_and_draw()
    plt.show()


def save_snapshot(path: str, p: SimulationParams) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    status_text = fig.text(0.06, 0.03, "", fontsize=10)
    draw_scene(ax, status_text, p)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    params = apply_constraints(
        SimulationParams(
            t0=args.t0,
            t=args.t,
            t_max=args.t_max,
            v=args.v,
            dv=args.delta_v,
            delta_t=args.delta_t,
            t_acc=args.t_acc,
            x0=args.x0,
            c=args.c,
        )
    )

    if args.output:
        save_snapshot(args.output, params)

    if not args.no_show:
        build_ui(params)


if __name__ == "__main__":
    main()
