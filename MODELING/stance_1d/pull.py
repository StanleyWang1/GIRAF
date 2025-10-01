import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Config ---
N_TRIALS = 100000
MAX_PULL = 60  # guaranteed at 60
UPDATE_EVERY = 500  # batch updates for speed

def pull_prob(pull_number: int) -> float:
    if 1 <= pull_number <= 10:  return 0.01
    if 11 <= pull_number <= 20: return 0.012
    if 21 <= pull_number <= 30: return 0.015
    if 31 <= pull_number <= 40: return 0.018
    if 41 <= pull_number <= 50: return 0.025
    if 51 <= pull_number <= 59: return 0.60
    if pull_number == 60:       return 1.0
    raise ValueError("Pull number out of range 1..60")

def simulate_one_trial() -> int:
    for k in range(1, MAX_PULL + 1):
        if np.random.random() < pull_prob(k):
            return k
    return MAX_PULL

# Pre-simulate results (still revealed gradually in the animation)
results = np.array([simulate_one_trial() for _ in range(N_TRIALS)], dtype=int)

# Prepare evolving PDF
counts = np.zeros(MAX_PULL, dtype=int)
x = np.arange(1, MAX_PULL + 1)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(x, np.zeros_like(x, dtype=float), width=0.9, align='center')
ax.set_xlim(0.5, MAX_PULL + 0.5)
ax.set_ylim(0, 0.12)  # will auto-raise if needed
ax.set_xlabel("Draw number when first obtained (k)")
ax.set_ylabel("Probability mass (empirical)")
title = ax.set_title(f"Evolving PDF of Draws Needed (0 / {N_TRIALS} trials)")

def update(frame_idx):
    end = min((frame_idx + 1) * UPDATE_EVERY, N_TRIALS)
    new_chunk = results[frame_idx * UPDATE_EVERY : end]
    for k in new_chunk:
        counts[k - 1] += 1

    total = counts.sum()
    pdf = counts / total if total > 0 else counts.astype(float)

    # Grow y-limit if any bar nears top
    m = pdf.max() if total > 0 else 0.1
    if m > ax.get_ylim()[1] * 0.95:
        ax.set_ylim(0, m * 1.1)

    for rect, h in zip(bars, pdf):
        rect.set_height(h)

    title.set_text(f"Evolving PDF of Draws Needed ({total} / {N_TRIALS} trials)")
    return (*bars, title)

frames = (N_TRIALS + UPDATE_EVERY - 1) // UPDATE_EVERY
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False, repeat=False)

# Optional: save GIF
# anim.save("evolving_pdf.gif", writer=PillowWriter(fps=20))

plt.show()

# Final stats
mean_draws = results.mean()
print(f"Empirical mean draws needed after {N_TRIALS} trials: {mean_draws:.4f}")
