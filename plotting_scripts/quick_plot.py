import matplotlib.pyplot as plt

# --- Data ---
hat_bleu = [30.18, 31.68, 33.69, 33.61, 33.58]
hat_latency = [48.6, 79.3, 110.8, 143.5, 175.5]

ours_bleu = [30.99, 32.94, 33.47, 33.69, 33.57]
ours_latency = [50.413, 79.431, 110.073, 143.125, 176.379]

reported_bleu = [33.4, 34.2, 34.5, 34.7, 34.8]
reported_latency = [45.6, 74.5, 109.0, 137.8, 168.8]
# --- Plot ---
plt.figure(figsize=(6,4))
plt.plot(hat_latency, hat_bleu, 'o-', color='#c44', label='HAT Calculated')
plt.plot(ours_latency, ours_bleu, 's-', color='#06c', label='Ours')
plt.plot(reported_latency, reported_bleu, 's-', color='#2A1', label='HAT Paper')

plt.xlabel("Latency (ms)")
plt.ylabel("BLEU Score")
plt.title("BLEU vs. Latency Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pareto_compare.png', dpi=200)