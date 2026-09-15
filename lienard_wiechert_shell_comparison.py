# lw_visualiser.py
# Exact Liénard–Wiechert accelerated charge visualiser
# pip install numpy matplotlib
#
# Features:
# - exact retarded-time solve (bisection)
# - exact LW E-field in 2D slice
# - streamlines
# - spacetime-consistent causal shells
# - toggle exact/nonconcentric vs lecture concentric shell
# - slider time control
#
# Physics sources checked:
# PyCharge / moving-point-charges / standard LW equations
# https://github.com/MatthewFilipovich/pycharge  [oai_citation:0‡GitHub](https://github.com/MatthewFilipovich/pycharge?utm_source=chatgpt.com)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# =========================================================
# USER PARAMETERS
# =========================================================

c = 1.0
q = 1.0

t_acc = 2.0          # acceleration start
dt_acc = 1.0         # duration
t_end = t_acc + dt_acc

v0 = 0.25            # initial beta=v/c
v1 = 0.75            # final beta=v/c

GRID_X = (-8, 14)
GRID_Y = (-7, 7)
NX = 240
NY = 180

FIELD_DENSITY = 1.7
LINEWIDTH = 0.8

USE_EXACT_SHELLS = True   # False = lecturer concentric cΔt shell mode

# =========================================================
# MOTION MODEL
# =========================================================

a = (v1 - v0) / dt_acc

def x_of_t(t):
    t = np.asarray(t)
    out = np.empty_like(t, dtype=float)

    before = t <= t_acc
    during = (t > t_acc) & (t <= t_end)
    after  = t > t_end

    out[before] = v0 * t[before]

    tau = t[during] - t_acc
    x_start = v0 * t_acc
    out[during] = x_start + v0*tau + 0.5*a*tau**2

    x_end = x_start + v0*dt_acc + 0.5*a*dt_acc**2
    out[after] = x_end + v1*(t[after]-t_end)

    return out

def v_of_t(t):
    t = np.asarray(t)
    out = np.empty_like(t, dtype=float)

    out[t <= t_acc] = v0

    mid = (t > t_acc) & (t <= t_end)
    out[mid] = v0 + a*(t[mid]-t_acc)

    out[t > t_end] = v1
    return out

def acc_of_t(t):
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    out[(t > t_acc) & (t <= t_end)] = a
    return out

# =========================================================
# RETARDED TIME SOLVER
# =========================================================

def retarded_time(X, Y, t_obs, n_iter=55):
    lo = np.full_like(X, -40.0)
    hi = np.full_like(X, t_obs)

    for _ in range(n_iter):
        mid = 0.5*(lo+hi)

        xs = x_of_t(mid)
        R = np.sqrt((X-xs)**2 + Y**2)

        f = mid + R/c - t_obs

        mask = f > 0
        hi = np.where(mask, mid, hi)
        lo = np.where(mask, lo, mid)

    return 0.5*(lo+hi)

# =========================================================
# EXACT LIENARD-WIECHERT FIELD
# =========================================================

def lw_field(X, Y, t_obs):

    tr = retarded_time(X, Y, t_obs)

    xs = x_of_t(tr)
    beta = v_of_t(tr)
    betadot = acc_of_t(tr)

    Rx = X - xs
    Ry = Y
    R = np.sqrt(Rx**2 + Ry**2)
    R = np.maximum(R, 1e-5)

    nx = Rx / R
    ny = Ry / R

    kappa = 1.0 - nx*beta
    kappa = np.maximum(kappa, 1e-5)

    gamma2_inv = 1.0 - beta**2

    # velocity field ~1/R^2
    Evx = q * gamma2_inv * (nx - beta) / (kappa**3 * R**2)
    Evy = q * gamma2_inv * ny         / (kappa**3 * R**2)

    # acceleration field ~1/R
    ndotbd = nx * betadot
    ndotnm = 1.0 - nx*beta

    Eax = q * ((nx-beta)*ndotbd - betadot*ndotnm) / (kappa**3 * R)
    Eay = q * (ny*ndotbd) / (kappa**3 * R)

    Ex = Evx + Eax
    Ey = Evy + Eay

    return Ex, Ey

# =========================================================
# DRAW HELPERS
# =========================================================

def draw_circle(ax, xc, yc, r, **kw):
    if r <= 0:
        return
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(xc + r*np.cos(th), yc + r*np.sin(th), **kw)

def shell_positions():
    x0 = x_of_t(np.array([0.0]))[0]
    xs = x_of_t(np.array([t_acc]))[0]
    xe = x_of_t(np.array([t_end]))[0]
    return x0, xs, xe

# =========================================================
# GRID
# =========================================================

xs = np.linspace(*GRID_X, NX)
ys = np.linspace(*GRID_Y, NY)
X, Y = np.meshgrid(xs, ys)

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(10,8))
plt.subplots_adjust(bottom=0.18)

sax = plt.axes([0.18, 0.07, 0.60, 0.03])
slider = Slider(sax, "time", 0.15, 10.0, valinit=6.0, valstep=0.03)

bax = plt.axes([0.81, 0.05, 0.12, 0.06])
btn = Button(bax, "toggle shell")

# =========================================================
# UPDATE
# =========================================================

def redraw(t):

    ax.clear()

    Ex, Ey = lw_field(X, Y, t)

    mag = np.sqrt(Ex**2 + Ey**2)
    Exn = Ex / (mag + 1e-9)
    Eyn = Ey / (mag + 1e-9)

    lw = 0.3 + 1.8*np.tanh(mag/np.percentile(mag, 85))

    ax.streamplot(
        xs, ys, Exn, Eyn,
        density=FIELD_DENSITY,
        linewidth=LINEWIDTH,
        arrowsize=0.7
    )

    # current positions
    x_actual = x_of_t(np.array([t]))[0]
    x_unacc = v0 * t

    ax.scatter([x_unacc], [0], s=55, color="royalblue", zorder=5, label="unaccelerated")
    ax.scatter([x_actual], [0], s=85, color="orange", zorder=6, label="actual charge")

    # shell circles
    x_init, x_start, x_end = shell_positions()

    if USE_EXACT_SHELLS:
        draw_circle(ax, x_start, 0, c*(t-t_acc), color="green", lw=2, alpha=0.85)
        draw_circle(ax, x_end,   0, c*(t-t_end), color="red", lw=2, alpha=0.85)
        shelltxt = "Exact shells (non-concentric)"
    else:
        # lecture mode: same centre, thickness cΔt
        r_outer = c*(t-t_acc)
        r_inner = c*(t-t_end)
        centre = x_start
        draw_circle(ax, centre, 0, r_outer, color="green", lw=2, alpha=0.85)
        draw_circle(ax, centre, 0, r_inner, color="red", lw=2, alpha=0.85)
        shelltxt = "Far-field lecture shells (concentric)"

    # initial signal cone
    draw_circle(ax, x_init, 0, c*t, color="black", ls=":", lw=1.0, alpha=0.5)

    # labels
    ax.text(0.02, 0.98,
            f"{shelltxt}\nβ0={v0:.2f}, β1={v1:.2f}\nt={t:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(fc="white", alpha=0.8))

    ax.set_xlim(*GRID_X)
    ax.set_ylim(*GRID_Y)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title("Exact Liénard–Wiechert Accelerating Charge Visualiser")
    ax.legend(loc="upper right")

    fig.canvas.draw_idle()

def slider_update(val):
    redraw(val)

def toggle(event):
    global USE_EXACT_SHELLS
    USE_EXACT_SHELLS = not USE_EXACT_SHELLS
    redraw(slider.val)

slider.on_changed(slider_update)
btn.on_clicked(toggle)

redraw(slider.val)
plt.show()