import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
import os

# ==========================================
# 1. SYSTEM & ACTUATOR GEOMETRY
# ==========================================
# Hopf Normal Form: Open-loop has an unstable origin and a stable limit cycle at r = sqrt(MU)
MU = 0.5     
OMEGA = 1.0  # Natural frequency
W = 1.0      # Reference frequency (matches natural)
R = 1.3      # Reference radius (larger than natural limit cycle)
K_P = 5.0    # Proportional tracking gain

# Control Effectiveness Matrix B (3 actuators, 2D force space)
B = np.array([
    [1.0, -0.5, -0.5],
    [0.0,  0.866, -0.866]
])
U_MAX = 1.0

def get_ams_vertices(B_mat, alpha_vec):
    """Compute the Attainable Moment Set (AMS) polygon vertices."""
    B_eff = B_mat * alpha_vec
    vertices = []
    for u1 in [-U_MAX, U_MAX]:
        for u2 in [-U_MAX, U_MAX]:
            for u3 in [-U_MAX, U_MAX]:
                u = np.array([u1, u2, u3])
                v = B_eff @ u
                vertices.append(v)
    vertices = np.array(vertices)
    hull = ConvexHull(vertices)
    return vertices[hull.vertices]

# Healthy vs Degraded (Actuator 1 loses 80% effectiveness)
alpha_healthy = np.array([1.0, 1.0, 1.0])
alpha_degraded = np.array([0.2, 1.0, 1.0])

ams_healthy = get_ams_vertices(B, alpha_healthy)
ams_degraded = get_ams_vertices(B, alpha_degraded)

# ==========================================
# 2. CLOSED-LOOP DYNAMICS WITH SATURATION
# ==========================================
def ref_func(t):
    return R * np.cos(W * t), R * np.sin(W * t)

def closed_loop_dynamics(t, state, alpha_vec):
    x, y = state
    x_r, y_r = ref_func(t)
    
    # Exact Feedforward (cancels plant dynamics at reference)
    # v_ff = dot(x_r) - f(x_r)
    v_ff_x = (R**2 - MU) * x_r
    v_ff_y = (R**2 - MU) * y_r
    
    # Proportional Feedback
    v_cmd_x = v_ff_x - K_P * (x - x_r)
    v_cmd_y = v_ff_y - K_P * (y - y_r)
    v_cmd = np.array([v_cmd_x, v_cmd_y])
    
    # Control Allocation (Pseudo-inverse)
    B_eff = B * alpha_vec
    B_pinv = np.linalg.pinv(B_eff)
    u_cmd = B_pinv @ v_cmd
    
    # Physical Actuator Saturation (The Nonlinearity)
    u_act = np.clip(u_cmd, -U_MAX, U_MAX)
    
    # Delivered Virtual Control
    v_act = B_eff @ u_act
    
    # Plant Dynamics (Hopf Normal Form)
    r2 = x**2 + y**2
    dx = MU * x - OMEGA * y - x * r2 + v_act[0]
    dy = OMEGA * x + MU * y - y * r2 + v_act[1]
    
    return [dx, dy]

# ==========================================
# 3. SIMULATION
# ==========================================
t_span = (0, 60) # Longer time to show asymptotic behavior
t_eval = np.linspace(t_span[0], t_span[1], 3000)

sol_healthy = solve_ivp(closed_loop_dynamics, t_span, [0.1, 0.1], 
                        args=(alpha_healthy,), t_eval=t_eval)
sol_degraded = solve_ivp(closed_loop_dynamics, t_span, [0.1, 0.1], 
                         args=(alpha_degraded,), t_eval=t_eval)

# ==========================================
# 4. PLOTTING THE DELIVERABLE
# ==========================================
os.makedirs("results", exist_ok=True)
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: The Attainable Moment Set (AMS) Geometry
axs[0].fill(ams_healthy[:, 0], ams_healthy[:, 1], alpha=0.15, color='green', label='Healthy AMS')
axs[0].plot(np.append(ams_healthy[:, 0], ams_healthy[0,0]), 
            np.append(ams_healthy[:, 1], ams_healthy[0,1]), 'g-', linewidth=2)

axs[0].fill(ams_degraded[:, 0], ams_degraded[:, 1], alpha=0.2, color='red', label='Degraded AMS (α_1=0.2)')
axs[0].plot(np.append(ams_degraded[:, 0], ams_degraded[0,0]), 
            np.append(ams_degraded[:, 1], ams_degraded[0,1]), 'r--', linewidth=2)

# Plot the demanded feedforward virtual control trajectory (a circle)
theta = np.linspace(0, 2*np.pi, 200)
v_ff_mag = (R**2 - MU) * R
v_ff_x = v_ff_mag * np.cos(theta)
v_ff_y = v_ff_mag * np.sin(theta)
axs[0].plot(v_ff_x, v_ff_y, 'k-', linewidth=2, label=f'Demanded $v_{{ff}}$ (Mag={v_ff_mag:.2f})')

axs[0].set_title("Control Effectiveness & Attainable Moment Set (AMS)\nDegradation shrinks the AMS asymmetrically", fontsize=14)
axs[0].set_xlabel("Virtual Control $v_1$ (Force X)", fontsize=12)
axs[0].set_ylabel("Virtual Control $v_2$ (Force Y)", fontsize=12)
axs[0].legend(fontsize=11, loc='upper right')
axs[0].grid(True)
axs[0].set_xlim(-2.5, 2.5)
axs[0].set_ylim(-2.5, 2.5)
axs[0].set_aspect('equal')

# Plot 2: State Space Bifurcation
# Reference trajectory
ref_x = R * np.cos(W * t_eval)
ref_y = R * np.sin(W * t_eval)
axs[1].plot(ref_x, ref_y, 'k--', alpha=0.6, linewidth=2, label=f'Reference Trajectory (R={R})')

# Healthy tracks perfectly
axs[1].plot(sol_healthy.y[0], sol_healthy.y[1], 'g-', alpha=0.8, linewidth=2, label='Healthy: Perfect Tracking')

# Degraded falls to natural limit cycle
axs[1].plot(sol_degraded.y[0], sol_degraded.y[1], 'r-', alpha=0.8, linewidth=2, label='Degraded: Attractor Shift')

# Plot natural limit cycle for context
nat_r = np.sqrt(MU)
nat_theta = np.linspace(0, 2*np.pi, 100)
axs[1].plot(nat_r * np.cos(nat_theta), nat_r * np.sin(nat_theta), 'b:', alpha=0.5, linewidth=2, label=f'Open-Loop Attractor (r={nat_r:.2f})')

axs[1].set_title("State-Space Phase Portrait\nSaturation destroys the tracked attractor, system falls to open-loop limit cycle", fontsize=14)
axs[1].set_xlabel("State $x_1$", fontsize=12)
axs[1].set_ylabel("State $x_2$", fontsize=12)
axs[1].legend(fontsize=11, loc='upper right')
axs[1].grid(True)
axs[1].set_aspect('equal')
axs[1].set_xlim(-2, 2)
axs[1].set_ylim(-2, 2)

plt.tight_layout()
plt.savefig("results/partial_redundancy_ams_bifurcation.png", dpi=300)
plt.close()

