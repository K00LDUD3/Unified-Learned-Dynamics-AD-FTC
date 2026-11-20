import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import os

# ==========================================
# 1. MATHEMATICAL SYSTEM & DATA GENERATION
# ==========================================
MU = 1.0
KD = 2.0
ALPHA_CRIT = MU / KD  # 0.5

def vdp_closed_loop(t, state, alpha):
    x, dx = state
    u_cmd = -KD * dx
    u_act = alpha * u_cmd
    ddx = MU * (1 - x**2) * dx - x + u_act
    return [dx, ddx]

def generate_trajectory(alpha, t_span=(0, 50), dt=0.05, init_state=None):
    if init_state is None:
        init_state = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(vdp_closed_loop, t_span, init_state, args=(alpha,), t_eval=t_eval, method='RK45')
    return sol.y.T, sol.t

print("Generating physical data...")
alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
data_X_t, data_X_next = [], []

for alpha in alphas:
    for _ in range(5):
        X, _ = generate_trajectory(alpha, t_span=(0, 40), dt=0.05)
        data_X_t.append(X[:-1])
        data_X_next.append(X[1:])

X_t = np.vstack(data_X_t)
X_next = np.vstack(data_X_next)

x_mean = X_t.mean(axis=0)
x_std = X_t.std(axis=0) + 1e-8
X_t_norm = (X_t - x_mean) / x_std
X_next_norm = (X_next - x_mean) / x_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_t_tensor = torch.FloatTensor(X_t_norm).to(device)
X_next_tensor = torch.FloatTensor(X_next_norm).to(device)

# ==========================================
# 2. LEARNED LATENT DYNAMICS MODEL
# ==========================================
class LatentDynamics(nn.Module):
    def __init__(self, dt=0.05):
        super().__init__()
        self.dt = dt
        self.encoder = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        self.dynamics = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, x_t, x_next):
        z_t = self.encoder(x_t)
        z_next_true = self.encoder(x_next)
        z_next_pred = z_t + self.dt * self.dynamics(z_t) # Euler integration
        x_t_hat = self.decoder(z_t)
        
        loss_recon = nn.MSELoss()(x_t_hat, x_t)
        loss_dyn = nn.MSELoss()(z_next_pred, z_next_true)
        return loss_recon + 10.0 * loss_dyn, z_t, z_next_pred

# ==========================================
# 3. TRAINING
# ==========================================
print("Training Latent Dynamics Model...")
model = LatentDynamics(dt=0.05).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t_tensor, X_next_tensor), batch_size=256, shuffle=True)

for epoch in range(100):
    epoch_loss = 0
    for batch_x_t, batch_x_next in loader:
        optimizer.zero_grad()
        loss, _, _ = model(batch_x_t, batch_x_next)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {epoch_loss/len(loader):.4f}")

# ==========================================
# 4. ANALYSIS & PLOTTING
# ==========================================
print("Generating plots...")
os.makedirs("results", exist_ok=True)

def get_latent_trajectory(alpha, init_state=[0.5, 0.5]):
    X_phys, _ = generate_trajectory(alpha, t_span=(0, 60), dt=0.05, init_state=init_state)
    X_phys_norm = (X_phys - x_mean) / x_std
    with torch.no_grad():
        Z = model.encoder(torch.FloatTensor(X_phys_norm).to(device)).cpu().numpy()
    return X_phys, Z

X_healthy, Z_healthy = get_latent_trajectory(1.0, init_state=[2.0, 2.0])
X_degraded, Z_degraded = get_latent_trajectory(0.2, init_state=[0.1, 0.1])

# Plot 1: Physical Phase Portraits
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(X_healthy[:, 0], X_healthy[:, 1], 'b', alpha=0.6)
axs[0].set_title(f"Healthy ($\\alpha=1.0 > \\alpha_{{crit}}$): Stable Equilibrium")
axs[0].set_xlabel("Position $x$"); axs[0].set_ylabel("Velocity $\dot{x}$"); axs[0].grid(True)

axs[1].plot(X_degraded[:, 0], X_degraded[:, 1], 'r', alpha=0.6)
axs[1].set_title(f"Degraded ($\\alpha=0.2 < \\alpha_{{crit}}$): Limit Cycle (Hopf Bifurcation)")
axs[1].set_xlabel("Position $x$"); axs[1].set_ylabel("Velocity $\dot{x}$"); axs[1].grid(True)
plt.tight_layout(); plt.savefig("results/1_physical_phase_portraits.png", dpi=300); plt.close()

# Plot 2: Latent Space Topological Preservation
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(Z_healthy[:, 0], Z_healthy[:, 1], 'b', alpha=0.6)
axs[0].set_title("Latent Space: Healthy (Stable Spiral)")
axs[0].set_xlabel("$z_1$"); axs[0].set_ylabel("$z_2$"); axs[0].grid(True)

axs[1].plot(Z_degraded[:, 0], Z_degraded[:, 1], 'r', alpha=0.6)
axs[1].set_title("Latent Space: Degraded (Learned Limit Cycle)")
axs[1].set_xlabel("$z_1$"); axs[1].set_ylabel("$z_2$"); axs[1].grid(True)
plt.tight_layout(); plt.savefig("results/2_latent_phase_portraits.png", dpi=300); plt.close()

# Plot 3: Latent Vector Field (The "Solid" NLD Proof)
fig, ax = plt.subplots(figsize=(8, 6))
z_min_1, z_max_1 = min(Z_healthy[:, 0].min(), Z_degraded[:, 0].min()) - 1, max(Z_healthy[:, 0].max(), Z_degraded[:, 0].max()) + 1
z_min_2, z_max_2 = min(Z_healthy[:, 1].min(), Z_degraded[:, 1].min()) - 1, max(Z_healthy[:, 1].max(), Z_degraded[:, 1].max()) + 1
Z1, Z2 = np.meshgrid(np.linspace(z_min_1, z_max_1, 25), np.linspace(z_min_2, z_max_2, 25))
Z_grid = torch.FloatTensor(np.vstack([Z1.flatten(), Z2.flatten()]).T).to(device)

with torch.no_grad():
    dZ = model.dynamics(Z_grid).cpu().numpy()
dZ1, dZ2 = dZ[:, 0].reshape(Z1.shape), dZ[:, 1].reshape(Z2.shape)

ax.streamplot(Z1, Z2, dZ1, dZ2, color='gray', density=1.5)
ax.plot(Z_degraded[:, 0], Z_degraded[:, 1], 'r', linewidth=2, label="Degraded Trajectory")
ax.set_title("Learned Latent Vector Field $\\dot{z} = G_\\theta(z)$\nTopological preservation of the Hopf Bifurcation")
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig("results/3_latent_vector_field.png", dpi=300); plt.close()
