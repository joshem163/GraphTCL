# import numpy as np
# import matplotlib.pyplot as plt
#
# # ================================
# # 1. Enter your results here:
# # ================================
# results = {
#     "MUTAG": [89.12, 88.30, 87.95],
#     "BZR": [85.00, 84.40, 83.70],
#     "PTC": [71.52, 71.10, 70.90],
#     "COX2": [80.81, 80.30, 79.85],
#     "PROTEINS": [77.28, 76.75, 76.30],
#     "IMDB-B": [72.10, 71.92, 71.50],
#     "IMDB-M": [50.80, 50.21, 49.80]
# }
#
# filtrations = ["HKS", "Degree", "Closeness"]
#
#
# # ================================
# # 2. Radar Plot Function
# # ================================
# def make_radar_plot(dataset_name, values):
#     num_vars = len(filtrations)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     values = values + values[:1]  # repeat first value to close circle
#     angles += angles[:1]  # repeat first angle to close circle
#
#     fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
#
#     # Draw outline
#     ax.plot(angles, values, linewidth=2, linestyle='solid')
#     ax.fill(angles, values, alpha=0.25)
#
#     # Add labels
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(filtrations, fontsize=12)
#
#     # Set title
#     ax.set_title(f"{dataset_name}: Filtration Robustness", fontsize=14, pad=20)
#
#     # Value range (you can adjust)
#     ax.set_rlabel_position(30)
#     ax.set_ylim(min(values) - 2, max(values) + 2)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # ================================
# # 3. Generate for all datasets
# # ================================
# for dataset, vals in results.items():
#     make_radar_plot(dataset, vals)

# import numpy as np
# import matplotlib.pyplot as plt
#
# # ================================
# # 1. Enter your accuracies here
# # ================================
# results = {
#     "MUTAG":     [89.12, 88.30, 87.95],
#     "BZR":       [85.00, 84.40, 83.70],
#     "PTC":       [71.52, 71.10, 70.90],
#     "COX2":      [80.81, 80.30, 79.85],
#     "PROTEINS":  [77.28, 76.75, 76.30],
#     "IMDB-B":    [72.10, 71.92, 71.50],
#     "IMDB-M":    [50.80, 50.21, 49.80]
# }
#
# filtrations = ["HKS", "Degree", "Closeness"]
# N = len(filtrations)
#
# # ================================
# # 2. Prepare radar axes
# # ================================
# angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# angles += angles[:1]        # close the loop
#
# # ================================
# # 3. Create combined radar plot
# # ================================
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, polar=True)
#
# # Colors for datasets
# colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
#
# for (dataset, vals), color in zip(results.items(), colors):
#     stats = vals + vals[:1]  # close loop
#     ax.plot(angles, stats, linewidth=2, label=dataset, color=color)
#     ax.fill(angles, stats, alpha=0.15, color=color)
#
# # ================================
# # 4. Formatting
# # ================================
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(filtrations, fontsize=14)
#
# # Adjust radial limits nicely
# min_val = min(min(v) for v in results.values()) - 3
# max_val = max(max(v) for v in results.values()) + 3
# ax.set_ylim(min_val, max_val)
#
# # Title
# ax.set_title("Model Stability Across Filtrations", fontsize=16, pad=20)
#
# # Legend
# plt.legend(bbox_to_anchor=(1.1, 1.1), fontsize=10)
#
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1. Enter Your Results Here
# ================================
results = {
    "MUTAG":     [95.73, 94.71, 95.23],
    "BZR":       [92.85, 93.32, 92.73],
    "PTC":       [74.97, 73.55, 74.12],
    "COX2":      [88.00, 88.67, 89.11],
    "PROTEINS":  [79.16, 79.70, 79.79],
    "IMDB-B":    [75.90, 75.00, 74.90],
    "IMDB-M":    [51.20, 52.00, 51.00]
}

# Format: [HKS, Degree, Closeness]
datasets = list(results.keys())

delta_degree = []
delta_closeness = []

for ds in datasets:
    hks, deg, clo = results[ds]
    delta_degree.append(deg - hks)
    delta_closeness.append(clo - hks)

# ================================
# 2. Plot Relative Drop
# ================================
x = np.arange(len(datasets))
width = 0.35

plt.figure(figsize=(10,5))
plt.axhline(0, color="black", linewidth=1)

# plt.bar(x - width/2, delta_degree, width, label="Degree - HKS")
# plt.bar(x + width/2, delta_closeness, width, label="Closeness - HKS")

plt.bar(
    x - width/2,
    delta_degree,
    width,
    label="Degree - HKS",
    color='navy'      # blue
)

plt.bar(
    x + width/2,
    delta_closeness,
    width,
    label="Closeness - HKS",
    color="#1ABC9C"     # orange
)

plt.xticks(x, datasets, rotation=30, fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("Δ Accuracy (%)", fontsize=16)
plt.title("Relative Drop Compared to HKS (Higher = Closer to HKS)", fontsize=16)
plt.legend(fontsize=16)

plt.tight_layout()
plt.savefig('relative_drop.pdf', format='pdf', dpi=300)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # ================================
# # Input Your Results Here
# # ================================
# results = {
#     "MUTAG":     [89.12, 88.30, 87.95],
#     "BZR":       [85.00, 84.40, 83.70],
#     "PTC":       [71.52, 71.10, 70.90],
#     "COX2":      [80.81, 80.30, 79.85],
#     "PROTEINS":  [77.28, 76.75, 76.30],
#     "IMDB-B":    [72.10, 71.92, 71.50],
#     "IMDB-M":    [50.80, 50.21, 49.80]
# }
#
# filtrations = ["HKS", "Degree", "Closeness"]
# x = np.arange(len(filtrations))
#
# plt.figure(figsize=(8,6))
#
# # Create one line per dataset
# for dataset, vals in results.items():
#     plt.plot(x, vals, marker='o', linewidth=2, label=dataset)
#
# plt.xticks(x, filtrations, fontsize=12)
# plt.ylabel("Accuracy (%)", fontsize=12)
# plt.title("Filtration Sensitivity of Our Model", fontsize=14)
# plt.legend(loc="best", fontsize=9)
#
# plt.tight_layout()
# plt.show()
