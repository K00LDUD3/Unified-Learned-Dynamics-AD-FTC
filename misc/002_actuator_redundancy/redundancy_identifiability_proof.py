import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==========================================
# 1. SYSTEM DYNAMICS & CONTROL LAW
# ==========================================
# Canonical Duffing Oscillator: ddot{x} + delta*dot{x} - x + x^3 = u_total
DELTA = 0.2
KP = 15.0
KD = 6.0

def get_reference(t):
    # Periodic orbit crossing the potential barrier
    w = 0.8
    A = 1.2
    x_ref = A * np.sin(w * t)
    dx_ref = A * w * np.cos(w * t)
    ddx_ref = -A * w**2 * np.sin(w * t)
    return x_ref, dx_ref, ddx_ref

def simulate_duffing(alpha1, alpha2, t_span=(0, 40), dt=0.01):
    def closed_loop_dynamics(t, state):
        x, dx = state
        x_ref, dx_ref, ddx_ref = get_reference(t)
        
        # Nominal Feedback Linearization + PD Control
        # u_cmd_total is the total force requested by the controller
        u_cmd_total = ddx_ref + DELTA * dx_ref - x_ref + x_ref**3 + KP * (x_ref - x) + KD * (dx_ref - dx)
        
        # Control Allocation (50/50 split, controller is blind to alpha)
        u_cmd_1 = 0.5 * u_cmd_total
        u_cmd_2 = 0.5 * u_cmd_total
        
        # Physical Actuator Effectiveness
        u_act_1 = alpha1 * u_cmd_1
        u_act_2 = alpha2 * u_cmd_2
        
        # Actual physical force delivered to the plant
        u_total_delivered = u_act_1 + u_act_2
        
        # Duffing dynamics: ddot{x} = -delta*dx + x - x^3 + u_total_delivered
        ddx = -DELTA * dx + x - x**3 + u_total_delivered
        return [dx, ddx]

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(closed_loop_dynamics, t_span, [1.0, 0.0], t_eval=t_eval, method='RK45')
    
    # Calculate actual actuator forces for plotting
    X = sol.y[0]
    DX = sol.y[1]
    U_cmd_total = np.zeros_like(X)
    for i, t in enumerate(sol.t):
        x_ref, dx_ref, ddx_ref = get_reference(t)
        U_cmd_total[i] = ddx_ref + DELTA * dx_ref - x_ref + x_ref**3 + KP * (x_ref - X[i]) + KD * (dx_ref - DX[i])
        
    u_act_1 = alpha1 * 0.5 * U_cmd_total
    u_act_2 = alpha2 * 0.5 * U_cmd_total
    
    return sol.t, sol.y, u_act_1, u_act_2

# ==========================================
# 2. RUN SCENARIOS
# ==========================================
print("Simulating redundant actuator scenarios...")

# Scenario A: Asymmetric Fault (Actuator 1 degrades to 50%, Actuator 2 is healthy)
# Total effectiveness on the commanded sum = 0.5*0.5 + 0.5*1.0 = 0.75
t_A, state_A, uA1, uA2 = simulate_duffing(0.5, 1.0)

# Scenario B: Symmetric Fault (Both actuators degrade to 75%)
# Total effectiveness on the commanded sum = 0.5*0.75 + 0.5*0.75 = 0.75
t_B, state_B, uB1, uB2 = simulate_duffing(0.75, 0.75)

# Scenario C: Alternative Asymmetric (Actuator 1 healthy, Actuator 2 degrades to 50%)
# Total effectiveness on the commanded sum = 0.5*1.0 + 0.5*0.5 = 0.75
t_C, state_C, uC1, uC2 = simulate_duffing(1.0, 0.5)

# Reference trajectory
t_ref = t_A
x_ref, _, _ = get_reference(t_ref)

# ==========================================
# 3. PLOTTING & DELIVERABLE
# ==========================================
os.makedirs("results", exist_ok=True)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Phase Portrait (State Space)
axs[0].plot(x_ref, np.gradient(x_ref, t_ref), 'k--', label='Reference Trajectory', alpha=0.5)
axs[0].plot(state_A[0], state_A[1], 'r', label='Fault A: α=[0.5, 1.0]', linewidth=2)
axs[0].plot(state_B[0], state_B[1], 'b', label='Fault B: α=[0.75, 0.75]', linewidth=2)
axs[0].plot(state_C[0], state_C[1], 'g', label='Fault C: α=[1.0, 0.5]', linewidth=2)
axs[0].set_title("State-Space Phase Portrait (Indistinguishable External Behavior)", fontsize=14)
axs[0].set_xlabel("Position $x$", fontsize=12)
axs[0].set_ylabel("Velocity $\dot{x}$", fontsize=12)
axs[0].legend(fontsize=11)
axs[0].grid(True)

# Plot 2: Internal Actuator Forces (The hidden reality)
axs[1].plot(t_A, uA1, 'r-', label='$u_{act,1}$ (Fault A)', alpha=0.7)
axs[1].plot(t_A, uA2, 'r--', label='$u_{act,2}$ (Fault A)', alpha=0.7)
axs[1].plot(t_B, uB1, 'b-', label='$u_{act,1}$ (Fault B)', alpha=0.7)
axs[1].plot(t_B, uB2, 'b--', label='$u_{act,2}$ (Fault B)', alpha=0.7)
axs[1].set_xlim([10, 20]) # Zoom in on steady state
axs[1].set_title("Internal Actuator Forces (Distinct Physical Realities)", fontsize=14)
axs[1].set_xlabel("Time $t$", fontsize=12)
axs[1].set_ylabel("Delivered Force $u_{act}$", fontsize=12)
axs[1].legend(fontsize=10, ncol=2)
axs[1].grid(True)

plt.tight_layout()
plt.savefig("results/redundancy_identifiability_wall.png", dpi=300)
plt.close()
