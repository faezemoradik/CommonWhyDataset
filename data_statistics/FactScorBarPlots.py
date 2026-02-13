import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Seaborn dark theme
# =========================
sns.set_theme(
    style="darkgrid",
    context="paper",
    font_scale=1.4,
)

# =========================
# Example data
# =========================
data = {
    "Model": ["GPT-5.1", "Gemini-2.5-Flash", "Llama-3.3-70B", "DeepSeek-V3.2", "OpenAI-o3"] * 2,
    "Method": ["Head"] * 5 + ["Long-Tail"] * 5,
    "Fact Score": [
        79.08, 71.66, 69.66, 70.95, 74.33,   # Full set
        68.35, 58.33, 41.99, 58.08, 58.42
    ]
}

df = pd.DataFrame(data)
errors = [0.05, 0.04, 0.08, 0.09, 0.05]

# =========================
# Plot
# =========================
plt.figure(figsize=(8.5, 4))

ax = sns.barplot(
    data=df,
    x="Model",
    y="Fact Score",
    hue="Method",
    palette=["tab:blue", "tab:orange"],
)

# =========================
# Add error bars only for the middle bar (Random SS)
# =========================
# In Seaborn barplots, patches are ordered by hue, then by x-category.
# Group 1 (Full set): patches 0-4
# Group 2 (Random SS): patches 5-9
# Group 3 (Our SS): patches 10-14
middle_bars = ax.patches[5:10]

for bar, error in zip(middle_bars, errors):
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()

    # ax.errorbar(
    #     x, y,
    #     yerr=error,
    #     fmt='none',
    #     c='black',
    #     capsize=4,
    #     linewidth=1.5
    # )

# Styling
ax.set_ylabel(r"FActScore ($\%$)")
ax.set_xlabel("")
ax.set_ylim(0, 100)
ax.legend(title="", frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig("Factscore.pdf", bbox_inches="tight")
plt.show()