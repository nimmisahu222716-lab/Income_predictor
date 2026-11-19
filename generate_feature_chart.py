import matplotlib.pyplot as plt
import numpy as np
import os

# Data
features = [
    'relationship',
    'workclass',
    'sex',
    'hours_per_week',
    'age',
    'capital_loss',
    'education',
    'capital_gain',
    'marital_status'
]

display_names = [
    'Relationship',
    'Workclass',
    'Sex',
    'Hours Per Week',
    'Age',
    'Capital Loss',
    'Education',
    'Capital Gain',
    'Marital Status'
]

importances = [0.08, 0.29, 0.44, 2.81, 4.85, 8.09, 17.34, 32.58, 33.52]

descriptions = [
    "Household role proxy reinforcing marital status and filing structure.",
    "Employment sector distinguishing government, private, and self-employed wages.",
    "Demographic factor capturing observed historical wage differentials.",
    "Work engagement metric separating part-time, standard, and overtime earners.",
    "Life-cycle career progression measuring workforce seniority and experience.",
    "Investment activity indicator reflecting capital assets and passive financial losses.",
    "Human capital proxy with progressive earning power across academic tiers.",
    "Primary investment indicator strongly associated with top tax brackets.",
    "Strongest single predictor distinguishing dual-income households from single earners."
]

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')

y_pos = np.arange(len(features))
colors = ['#B0BEC5', '#B0BEC5', '#90A4AE', '#78909C', '#607D8B', '#455A64', '#37474F', '#1E3A5F', '#0D233A']

bars = ax.barh(y_pos, importances, color=colors, height=0.65, edgecolor='#263238', linewidth=0.8)

# Add values and descriptions
for i, (bar, val, desc) in enumerate(zip(bars, importances, descriptions)):
    width = bar.get_width()
    # Percentage label
    ax.text(width + 0.6, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
            va='center', ha='left', fontsize=11, fontweight='bold', color='#1E293B')
    # One-line description
    ax.text(38.0, bar.get_y() + bar.get_height()/2, f"—  {desc}",
            va='center', ha='left', fontsize=10, color='#334155', style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, fontsize=12, fontweight='bold', color='#1E293B')
ax.set_xlabel('Gini Feature Importance (%)', fontsize=13, fontweight='bold', color='#1E293B', labelpad=12)
ax.set_xlim(0, 105)

# Style borders and grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')
ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
ax.set_axisbelow(True)

plt.title('Demographic Feature Importance Ranking & One-Line Descriptions (CART Gini Impurity)',
          fontsize=15, fontweight='bold', color='#0F172A', pad=20)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
out_path = 'outputs/feature_importance_horizontal.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved horizontal feature importance chart to {out_path}")
