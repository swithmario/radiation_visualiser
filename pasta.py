import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

c = 1.0
q = 1.0

t_acc = 2.0
dt_acc = 1.0
t_end = t_acc + dt_acc

v0 = 0.25
v1 = 0.65
a = (v1 - v0) / dt_acc

x0 = 0.0

def xq(t):
    t = np.asarray(t)
    x = np.empty_like(t, dtype=float)

    before = t <= t_acc
    during = (t > t_acc) & (t <= t_end)
    after = t > t_end

    x[before] = x0 + v0 * t[before]

    tau = t[during] - t_acc
    x_acc_start = x0 + v0 * t_acc
    x[during] = x_acc_start + v0 * tau + 0.5 * a * tau**2

    x_acc_end = x_acc_start + v0 * dt_acc + 0.5 * a * dt_acc**2
    x[after] = x_acc_end + v1 * (t[after] - t_end)

    return x

def vq(t):
    t = np.asarray(t)
    v = np.empty_like(t, dtype=float)
    v[t <= t_acc] = v0
    mid = (t > t_acc) & (t <= t_end)
    v[mid] = v0 + a * (t[mid] - t_acc)
    v[t > t_end] = v1
    return v

def aq(t):
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    out[(t > t_acc) & (t <= t_end)] = a
    return out

def retarded_time(X, Y, t_obs, n_iter=60):
    lo = np.full_like(X, -20.0)
    hi = np.full_like(X, t_obs)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        R = np.sqrt((X - xq(mid))**2 + Y**2)
        f = mid + R / c - t_obs

        hi = np.where(f > 0, mid, hi)
        lo = np.where(f <= 0, mid, lo)

    return 0.5 * (lo + hi)

def lienard_wiechert_E(X, Y, t_obs):
    tr = retarded_time(X, Y, t_obs)

    xs = xq(tr)
    beta_x = vq(tr) / c
    beta_dot_x = aq(tr) / c

    Rx = X - xs
    Ry = Y
    R = np.sqrt(Rx**2 + Ry**2)
    R = np.maximum(R, 1e-4)

    nx = Rx / R
    ny = Ry / R

    kappa = 1.0 - nx * beta_x
    kappa = np.maximum(kappa, 1e-4)

    beta2 = beta_x**2

    # velocity field:
    Evx = q * (nx - beta_x) * (1 - beta2) / (kappa**3 * R**2)
    Evy = q * ny * (1 - beta2) / (kappa**3 * R**2)

    # acceleration field:
    # n x ((n-beta) x beta_dot) in 2D, with beta_dot along x
    # vector triple product: (n-beta)(n·betadot) - betadot(n·(n-beta))
    ndot_bdot = nx * beta_dot_x
    ndot_n_minus_beta = 1.0 - nx * beta_x

    Eax = q * ((nx - beta_x) * ndot_bdot - beta_dot_x * ndot_n_minus_beta) / (c * kappa**3 * R)
    Eay = q * (ny * ndot_bdot) / (c * kappa**3 * R)

    return Evx + Eax, Evy + Eay

def draw_circle(ax, xc, r, **kw):
    if r <= 0:
        return
    th = np.linspace(0, 2*np.pi, 500)
    ax.plot(xc + r*np.cos(th), r*np.sin(th), **kw)

# grid
xs = np.linspace(-6, 9, 260)
ys = np.linspace(-6, 6, 220)
X, Y = np.meshgrid(xs, ys)

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.15)
sax = plt.axes([0.18, 0.05, 0.65, 0.03])
slider = Slider(sax, "t", 0.2, 9.0, valinit=6.7, valstep=0.02)

def update(t):
    ax.clear()

    Ex, Ey = lienard_wiechert_E(X, Y, t)
    mag = np.sqrt(Ex**2 + Ey**2)

    # cap extremes so streamplot behaves
    Ex = Ex / (mag + 1e-9)
    Ey = Ey / (mag + 1e-9)

    ax.streamplot(xs, ys, Ex, Ey, density=1.8, linewidth=0.8, arrowsize=0.7)

    # light-cone circles
    x_initial = xq(np.array([0.0]))[0]
    x_start = xq(np.array([t_acc]))[0]
    x_end = xq(np.array([t_end]))[0]

    draw_circle(ax, x_initial, c*(t-0.0), color="black", ls=":", lw=1.2, label="initial cone")
    draw_circle(ax, x_start, c*(t-t_acc), color="green", lw=1.8, label="acceleration starts")
    draw_circle(ax, x_end, c*(t-t_end), color="red", lw=1.8, label="acceleration ends")

    # particle positions
    ax.scatter([xq(np.array([t]))[0]], [0], s=80, color="orange", zorder=5, label="actual charge now")
    ax.scatter([x0 + v0*t], [0], s=60, color="blue", zorder=5, label="unaccelerated comparison")

    ax.set_aspect("equal")
    ax.set_xlim(-6, 9)
    ax.set_ylim(-6, 6)
    ax.grid(alpha=0.25)
    ax.set_title("Liénard–Wiechert electric field, retarded-time solved")
    ax.legend(loc="upper right", fontsize=8)
    fig.canvas.draw_idle()

slider.on_changed(update)
update(slider.val)
plt.show()