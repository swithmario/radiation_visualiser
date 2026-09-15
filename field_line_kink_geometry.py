#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Radiation Geometry for an Accelerated Charge
Recreated from scratch based on strict relativistic Lienard-Wiechert kinematics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def main():
    # Setup Figure and layout
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.35, top=0.92)
    
    # Define axes for interactive sliders
    ax_t    = plt.axes([0.15, 0.25, 0.65, 0.03])
    ax_v    = plt.axes([0.15, 0.20, 0.65, 0.03])
    ax_dv   = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_dt   = plt.axes([0.15, 0.10, 0.65, 0.03])
    ax_tacc = plt.axes([0.15, 0.05, 0.65, 0.03])

    # Sliders
    # Note: speed of light c = 1.0
    s_t    = Slider(ax_t, 'Time ($t$)', 0.0, 20.0, valinit=9.0)
    s_v    = Slider(ax_v, 'Velocity ($v$)', 0.0, 0.95, valinit=0.0)
    s_dv   = Slider(ax_dv, r'$\Delta v$', -0.9, 0.9, valinit=0.8)
    s_dt   = Slider(ax_dt, r'Duration ($\Delta t$)', 0.1, 3.0, valinit=0.8)
    s_tacc = Slider(ax_tacc, 'Start Accel ($t_{acc}$)', 0.0, 5.0, valinit=2.0)

    # Number of field lines to draw (determines flux density)
    N_lines = 32
    # Uniformly distributed in the rest frame to visualize "squashing" later
    theta_primes = np.linspace(0, 2 * np.pi, N_lines, endpoint=False)

    def update(val):
        t = s_t.val
        v0 = s_v.val
        dv = s_dv.val
        dt = s_dt.val
        t_acc = s_tacc.val

        # Constrain v1 so we never exceed or reach c (c=1.0)
        v1 = v0 + dv
        v1 = np.clip(v1, -0.999, 0.999)

        gamma0 = 1.0 / np.sqrt(1 - v0**2)
        gamma1 = 1.0 / np.sqrt(1 - v1**2)

        # Positions
        x_acc = v0 * t_acc
        # Approximate distance covered during brief acceleration phase
        x_end = x_acc + 0.5 * (v0 + v1) * dt 

        # Clear and configure main axis
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-10, 25)
        ax.set_ylim(-16, 16)
        ax.axhline(0, color='gray', linestyle=':', lw=1)
        ax.grid(True, linestyle=':', alpha=0.4)

        for tp in theta_primes:
            cos_tp = np.cos(tp)
            sin_tp = np.sin(tp)

            # 1. Outer field mapping (Pre-acceleration physics, velocity = v0)
            # Find the lab-frame angle phi0 of the photon emitted at t_acc
            cos_phi0 = (cos_tp + v0) / (1 + v0 * cos_tp)
            sin_phi0 = sin_tp / (gamma0 * (1 + v0 * cos_tp))
            
            # Electric field direction for v0
            E_dx_out = cos_phi0 - v0
            E_dy_out = sin_phi0

            # 2. Inner field mapping (Post-acceleration physics, velocity = v1)
            # Find the lab-frame angle phi1 of the photon emitted at t_acc + dt
            cos_phi1 = (cos_tp + v1) / (1 + v1 * cos_tp)
            sin_phi1 = sin_tp / (gamma1 * (1 + v1 * cos_tp))

            # Draw lines depending on time
            if t <= t_acc:
                # Still travelling at constant v0. No spheres exist yet.
                x_curr = v0 * t
                ax.plot([x_curr, x_curr + 50 * E_dx_out],[0, 50 * E_dy_out], 'k-', lw=1.2)

            elif t <= t_acc + dt:
                # Acceleration in progress. Only the outer (grander) sphere exists.
                R_out = t - t_acc
                P_out_x = x_acc + R_out * cos_phi0
                P_out_y = R_out * sin_phi0
                
                # Approximate particle location during acceleration
                frac = (t - t_acc) / dt
                v_curr = v0 + frac * (v1 - v0)
                x_curr = x_acc + 0.5 * (v0 + v_curr) * (t - t_acc)

                # Active Kink extending outwards
                ax.plot([x_curr, P_out_x], [0, P_out_y], color='red', lw=1.5)
                ax.plot([P_out_x, P_out_x + 50 * E_dx_out],[P_out_y, P_out_y + 50 * E_dy_out], 'k-', lw=1.2)

            else:
                # Acceleration complete. Both inner and outer spheres exist (Annulus).
                R_out = t - t_acc
                P_out_x = x_acc + R_out * cos_phi0
                P_out_y = R_out * sin_phi0
                
                R_in = t - (t_acc + dt)
                P_in_x = x_end + R_in * cos_phi1
                P_in_y = R_in * sin_phi1
                
                # Current location of the accelerated particle
                x_curr = x_end + v1 * (t - t_acc - dt)

                # The 3 segments of the field lines
                ax.plot([x_curr, P_in_x], [0, P_in_y], color='black', lw=1.2)        # Inside
                ax.plot([P_in_x, P_out_x],[P_in_y, P_out_y], color='red', lw=1.8)   # Kink across annulus
                ax.plot([P_out_x, P_out_x + 50 * E_dx_out],[P_out_y, P_out_y + 50 * E_dy_out], color='black', lw=1.2) # Outside

        # Draw the spheres marking the boundaries (The "Annulus")
        if t > t_acc:
            circle_out = plt.Circle((x_acc, 0), t - t_acc, color='blue', fill=False, linestyle='--', lw=1.2, alpha=0.6)
            ax.add_patch(circle_out)
        if t > t_acc + dt:
            circle_in = plt.Circle((x_end, 0), t - t_acc - dt, color='black', fill=False, linestyle='-', lw=1.2, alpha=0.5)
            ax.add_patch(circle_in)

        # Plot particle's current location
        if t <= t_acc:
            x_p = v0 * t
        elif t <= t_acc + dt:
            frac = (t - t_acc) / dt
            v_curr = v0 + frac * (v1 - v0)
            x_p = x_acc + 0.5 * (v0 + v_curr) * (t - t_acc)
        else:
            x_p = x_end + v1 * (t - t_acc - dt)

        ax.plot([x_p], [0], 'ko', markersize=7, zorder=5)

        # Update text
        ax.set_title(f"Radiation Geometry (t = {t:.2f})\n"
                     fr"$v_0 = {v0:.2f}$  ($\gamma_0 = {gamma0:.2f}$)  |  "
                     fr"$v_1 = {v1:.2f}$  ($\gamma_1 = {gamma1:.2f}$)", fontsize=13)

        fig.canvas.draw_idle()

    # Call update once to initialize the plot
    update(None)

    # Attach slider update events
    s_t.on_changed(update)
    s_v.on_changed(update)
    s_dv.on_changed(update)
    s_dt.on_changed(update)
    s_tacc.on_changed(update)

    # Adding a Reset button
    ax_reset = plt.axes([0.85, 0.05, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Reset')
    def reset(event):
        s_t.reset()
        s_v.reset()
        s_dv.reset()
        s_dt.reset()
        s_tacc.reset()
    btn_reset.on_clicked(reset)

    plt.show()

if __name__ == '__main__':
    main()