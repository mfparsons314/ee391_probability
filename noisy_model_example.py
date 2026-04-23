import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Style (large for slides)
# ----------------------------
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "lines.linewidth": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------------------
# Parameters
# ----------------------------
N = 200
noise_level = 0.8
seed = 0

np.random.seed(seed)

# ----------------------------
# Generate data
# ----------------------------
t = np.linspace(0, 4*np.pi, N)

true_signal = np.sin(t)
noise = noise_level * np.random.randn(N)
data = true_signal + noise

# Models
good_model = np.sin(t)
bad_model = np.sin(0.7 * t)  # wrong frequency

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# -------- Good model --------
ax = axes[0]

ax.fill_between(t, good_model, data, alpha=0.2)
ax.plot(t, data, alpha=0.6)
ax.plot(t, good_model)

ax.set_title("Good model")
ax.set_xlabel("t")
ax.set_ylabel("X(t)")

# -------- Bad model --------
ax = axes[1]

ax.fill_between(t, bad_model, data, alpha=0.2)
ax.plot(t, data, alpha=0.6)
ax.plot(t, bad_model)

ax.set_title("Bad model")
ax.set_xlabel("t")

# ----------------------------
# Layout
# ----------------------------
plt.tight_layout()
plt.savefig("model_fit_comparison.pdf")
plt.show()