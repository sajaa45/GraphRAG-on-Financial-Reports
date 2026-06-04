import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def to_pct_list(raw):
    return [None if v is None else 100 * v[0] / v[1] for v in raw]

def avg_pct(lst):
    vals = [v for v in lst if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# RAW DATA  (16 questions × 3 runs)
# ──────────────────────────────────────────────────────────────────────────────

# answer_source_traceability  (fractions; None = N/A)
ast_B_raw = [(14,14),(9,11),(5,5),(13,13),(2,2),(10,13),(1,4),(4,4),(9,10),(13,13),(11,12),(4,4),(9,9),(6,7),(13,14),(7,7)]
ast_A_raw = [(10,10),(6,6),(6,6),(11,11),(11,11),(9,9),(8,8),(3,4),(8,9),(6,7),(12,13),(4,6),(4,4),(7,7),(14,15),(9,12)]
ast_C_raw = [(12,12),(8,11),(5,5),(16,16),(2,2),(10,10),(2,2),(3,3),(9,9),(10,12),(12,12),(6,6),(8,8),(4,6),(15,15),(7,7)]

ast_A = to_pct_list(ast_A_raw)
ast_B = to_pct_list(ast_B_raw)
ast_C = to_pct_list(ast_C_raw)

# ──────────────────────────────────────────────────────────────────────────────
# AVERAGES
# ──────────────────────────────────────────────────────────────────────────────

avgs = {
    'Adapted Faithfulness': [avg_pct(ast_A), avg_pct(ast_B), avg_pct(ast_C)],
}

# ──────────────────────────────────────────────────────────────────────────────
# PALETTE
# ──────────────────────────────────────────────────────────────────────────────

COLORS  = ['#00b3ac', '#00b3ac', '#00b3ac']
DARKEN  = ['#c4661a', '#5a7249', '#007d78']   # edge / average line colors
LABELS  = ['Minerals Technologies', 'Kimco Realty', 'SolarWinds']

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE  –  Single chart for Source Traceability
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

x     = np.arange(len(LABELS))
width = 0.45
avg_vals = avgs['Adapted Faithfulness']

for i, (avg_v, col, ecol, lbl) in enumerate(
        zip(avg_vals, COLORS, DARKEN, LABELS)):
    bar = ax.bar(i, avg_v, width, color=col, edgecolor=ecol,
                 linewidth=0.7, alpha=0.90)
    ax.annotate(f'{avg_v:.1f}%',
                xy=(i, avg_v),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# pass threshold
ax.axhline(90, color='#d63031', linestyle='--', linewidth=1.1,
           label='Pass threshold (90%)')

ax.set_title('Adapted Faithfulness', fontsize=12, fontweight='bold', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel('Average Score (%)', fontsize=9)
ax.set_ylim(50, 108)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='gray', alpha=0.13, linewidth=0.5)
ax.legend(fontsize=8.5, framealpha=0.4, loc='lower right')

plt.tight_layout()
plt.savefig('eval_scores_per_metric.png', dpi=300, bbox_inches='tight')
plt.show()
print("Done.")