import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create high-res figure
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')

# Flow stages: (title, condition, profile_value, result, prob)
stages = [
    {
        "step": "STEP 1: Root Split",
        "feature": "Marital Status",
        "rule": "Is Marital Status == 'Married'?",
        "value": "Value: 'Married'",
        "decision": "YES (True Branch)",
        "prob": "P(>50K): 24.9% -> 44.7%",
        "color": "#1E3A5F"
    },
    {
        "step": "STEP 2: Investment Filter",
        "feature": "Capital Gain",
        "rule": "Is Capital Gain > $0?",
        "value": "Value: $5,000",
        "decision": "YES (True Branch)",
        "prob": "P(>50K): 44.7% -> 81.3%",
        "color": "#2C4A6F"
    },
    {
        "step": "STEP 3: Education Tier",
        "feature": "Education Level",
        "rule": "Education >= 'Graduation'?",
        "value": "Value: 'Graduation'",
        "decision": "YES (True Branch)",
        "prob": "P(>50K): 81.3% -> 89.2%",
        "color": "#3A5C80"
    },
    {
        "step": "STEP 4: Work Engagement",
        "feature": "Hours Per Week",
        "rule": "Hours Per Week >= 40?",
        "value": "Value: 45 hrs/week",
        "decision": "YES (True Branch)",
        "prob": "P(>50K): 89.2% -> 91.4%",
        "color": "#16A085"
    },
    {
        "step": "FINAL PREDICTION",
        "feature": "Terminal Leaf Node",
        "rule": "Class Assignment",
        "value": "Income > $50K",
        "decision": "HIGH CONFIDENCE",
        "prob": "Target: >50K (91.4%)",
        "color": "#0D5338"
    }
]

# Draw boxes
box_width = 2.2
box_height = 4.2
spacing = 0.5
start_x = 0.5
y_center = 3.0

for i, stage in enumerate(stages):
    x = start_x + i * (box_width + spacing)
    y = y_center - box_height / 2
    
    # Outer Card
    rect = patches.FancyBboxPatch((x, y), box_width, box_height,
                                  boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    
    # Header Banner
    header_rect = patches.FancyBboxPatch((x, y + box_height - 0.9), box_width, 0.9,
                                         boxstyle="round,pad=0.05,rounding_size=0.1",
                                         facecolor=stage["color"], edgecolor=stage["color"], zorder=3)
    ax.add_patch(header_rect)
    
    # Header Text
    ax.text(x + box_width/2, y + box_height - 0.35, stage["step"],
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF', zorder=4)
    ax.text(x + box_width/2, y + box_height - 0.65, stage["feature"],
            ha='center', va='center', fontsize=8.5, color='#E2E8F0', zorder=4)
    
    # Body Text
    ax.text(x + box_width/2, y + 2.5, f"Test Rule:\n{stage['rule']}",
            ha='center', va='center', fontsize=8.5, color='#334155', fontweight='medium', zorder=4)
    
    # Value Chip
    chip_rect = patches.FancyBboxPatch((x + 0.15, y + 1.6), box_width - 0.3, 0.5,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=0.8, zorder=3)
    ax.add_patch(chip_rect)
    ax.text(x + box_width/2, y + 1.85, stage["value"],
            ha='center', va='center', fontsize=8, color='#0F172A', fontweight='bold', zorder=4)
    
    # Outcome Tag
    ax.text(x + box_width/2, y + 1.15, stage["decision"],
            ha='center', va='center', fontsize=9, color='#059669' if 'YES' in stage['decision'] or 'HIGH' in stage['decision'] else '#DC2626',
            fontweight='bold', zorder=4)
    
    # Probability Badge
    prob_rect = patches.FancyBboxPatch((x + 0.1, y + 0.3), box_width - 0.2, 0.45,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor='#ECFDF5' if i == 4 else '#EFF6FF',
                                       edgecolor='#10B981' if i == 4 else '#3B82F6', linewidth=1.0, zorder=3)
    ax.add_patch(prob_rect)
    ax.text(x + box_width/2, y + 0.52, stage["prob"],
            ha='center', va='center', fontsize=8, color='#065F46' if i == 4 else '#1E40AF',
            fontweight='bold', zorder=4)
    
    # Arrow between stages
    if i < len(stages) - 1:
        arrow_start_x = x + box_width + 0.05
        arrow_end_x = arrow_start_x + spacing - 0.1
        ax.annotate('', xy=(arrow_end_x, y_center), xytext=(arrow_start_x, y_center),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=2.2, mutation_scale=15),
                    zorder=5)

ax.set_xlim(0, start_x + len(stages) * (box_width + spacing) + 0.2)
ax.set_ylim(0, 6)
ax.axis('off')

plt.title("Step-by-Step Traversal Walkthrough: Predicting Demographic Profile (P(>50K) = 91.4%)",
          fontsize=14, fontweight='bold', color='#0F172A', pad=15)

plt.tight_layout()
out_flow = 'outputs/step_by_step_decision_flow.png'
plt.savefig(out_flow, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved step by step flow diagram to {out_flow}")
