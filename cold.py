import numpy as np
import matplotlib.pyplot as plt

c = 1.0

# -----------------------------
# PARAMETERS
# -----------------------------
t0 = 0.0
x0 = 0.0

t_acc = 2.0
dt_acc = 1.0
t_end = t_acc + dt_acc
t_now = 7.0

v0 = 0.25
v1 = 0.65

assert abs(v0) < c
assert abs(v1) < c

# -----------------------------
# WORLDLINES
# -----------------------------
def x_unacc(t):
    return x0 + v0 * (t - t0)

x_acc_start = x_unacc(t_acc)
a = (v1 - v0) / dt_acc
x_acc_end = x_acc_start + v0 * dt_acc + 0.5 * a * dt_acc**2

def x_actual(t):
    t = np.asarray(t)
    x = np.empty_like(t, dtype=float)

    before = t <= t_acc
    during = (t > t_acc) & (t <= t_end)
    after = t > t_end

    x[before] = x0 + v0 * (t[before] - t0)

    tau = t[during] - t_acc
    x[during] = x_acc_start + v0 * tau + 0.5 * a * tau**2

    x[after] = x_acc_end + v1 * (t[after] - t_end)

    return x

# -----------------------------
# HELPERS
# -----------------------------
def draw_light_cone(ax, x_event, t_event, tmax, colour, label=None, ls="-", alpha=0.8):
    tt = np.linspace(t_event, tmax, 300)
    ax.plot(x_event + c * (tt - t_event), tt, color=colour, ls=ls, alpha=alpha, label=label)
    ax.plot(x_event - c * (tt - t_event), tt, color=colour, ls=ls, alpha=alpha)

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 7))

tt = np.linspace(t0, t_now, 1000)

ax.plot(x_unacc(tt), tt, "--", lw=2, color="tab:blue", label="hypothetical unaccelerated worldline")
ax.plot(x_actual(tt), tt, lw=4, color="tab:orange", label="actual accelerated worldline")

# key events
events = {
    "initial": (x0, t0),
    "actual accel starts": (x_acc_start, t_acc),
    "actual accel ends": (x_acc_end, t_end),
    "unacc at accel start": (x_unacc(t_acc), t_acc),
    "unacc at accel end": (x_unacc(t_end), t_end),
}

ax.scatter([x0], [t0], s=80, color="black", zorder=5)
ax.text(x0 + 0.05, t0 + 0.1, "initial position")

ax.scatter([x_acc_start], [t_acc], s=90, color="green", zorder=5)
ax.text(x_acc_start + 0.05, t_acc - 0.25, r"$t_{\rm acc}$ actual")

ax.scatter([x_acc_end], [t_end], s=90, color="red", zorder=5)
ax.text(x_acc_end + 0.05, t_end + 0.1, r"$t_{\rm acc}+\Delta t$ actual")

# current positions
x_unacc_now = x_unacc(t_now)
x_actual_now = x_actual(np.array([t_now]))[0]

ax.scatter([x_unacc_now], [t_now], s=100, color="tab:blue", zorder=5)
ax.text(x_unacc_now - 1.1, t_now - 0.35, "where charge\nwould be now")

ax.scatter([x_actual_now], [t_now], s=100, color="tab:orange", zorder=5)
ax.text(x_actual_now + 0.1, t_now - 0.35, "actual charge now")

# current time slice
ax.axhline(t_now, color="red", lw=2, alpha=0.5)
ax.text(-5.0, t_now + 0.08, "current time slice", color="red")

# light cones
draw_light_cone(ax, x0, t0, t_now, "black", "initial light cone", ls=":", alpha=0.9)

draw_light_cone(
    ax, x_unacc(t_acc), t_acc, t_now,
    "tab:blue", "unaccelerated cone from t_acc", ls="--", alpha=0.7
)

draw_light_cone(
    ax, x_unacc(t_end), t_end, t_now,
    "tab:cyan", "unaccelerated cone from t_acc+dt", ls="--", alpha=0.7
)

draw_light_cone(
    ax, x_acc_start, t_acc, t_now,
    "green", "actual cone: acceleration begins", ls="-", alpha=0.8
)

draw_light_cone(
    ax, x_acc_end, t_end, t_now,
    "red", "actual cone: acceleration ends", ls="-", alpha=0.8
)

# region labels
ax.text(4.1, 5.8, "outside first actual cone:\nold field still valid")
ax.text(4.9, 4.6, "between actual cones:\nradiation/kink shell")
ax.text(1.0, 5.7, "inside final actual cone:\nfield has updated")

ax.set_xlabel("space x")
ax.set_ylabel("time t")
ax.set_title("Spacetime diagram: accelerated charge and causal field update")
ax.grid(True)
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-5.5, 7.0)
ax.set_ylim(-0.3, t_now + 0.6)

plt.tight_layout()
plt.show()