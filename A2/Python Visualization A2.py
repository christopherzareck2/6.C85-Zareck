import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# ---------- Publication-quality styling ----------
# Changed: Integrated seaborn to easily enforce a clean, modern aesthetic
sns.set_theme(style="whitegrid", rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.4,
    'grid.color': '#DDDDDD'
})

# Refined color palette for colorblind safety and high contrast
COLORS = {
    'primary': '#2E86AB',   # Blue
    'secondary': '#A23B72', # Purplish red
    'tertiary': '#F18F01',  # Orange
    'quaternary': '#C73E1D',# Red
    'neutral': '#6C757D',   # Gray
}

# ---------- Load ----------
path = "Census_and_Corp_Ownership_and_Occupancy_Over_Time.csv"
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]

# ---------- Derived fields ----------
df = df.copy()
# Changed: Cleaned division-by-zero handling directly up front
tot_pop = df["tot_pop_all"].replace(0, np.nan)
df["pct_nonwhite"] = 1 - (df["white_all"] / tot_pop)
df["pct_black"] = df["black_all"] / tot_pop
df["pct_hisp"] = df["hisp_all"] / tot_pop
df["pct_aapi"] = df["aapi_all"] / tot_pop

# ---------- Figure 1: Density shifts ----------
# Changed: Replaced overlapping messy histograms with Density (KDE) plots.
years_to_compare = [2004, 2024]
df_compare = df[df["Year"].isin(years_to_compare)].copy()
df_compare["Year"] = df_compare["Year"].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Corporate ownership shift
sns.kdeplot(data=df_compare, x="corp_own_rate", hue="Year", fill=True, 
            common_norm=False, palette=[COLORS['primary'], COLORS['tertiary']], 
            alpha=0.5, linewidth=2, ax=axes[0])
axes[0].set_title("Shift in Corporate Ownership (2004 vs 2024)", fontweight='bold')
axes[0].set_xlabel("Corporate Ownership Rate")
axes[0].set_ylabel("Density")
axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Native percent formatting
sns.despine(ax=axes[0])

# Owner-occupancy shift
sns.kdeplot(data=df_compare, x="own_occ_rate", hue="Year", fill=True, 
            common_norm=False, palette=[COLORS['primary'], COLORS['tertiary']], 
            alpha=0.5, linewidth=2, ax=axes[1])
axes[1].set_title("Shift in Owner-Occupancy (2004 vs 2024)", fontweight='bold')
axes[1].set_xlabel("Owner-Occupancy Rate")
axes[1].set_ylabel("Density")
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig('fig1_distributions.png', dpi=300, bbox_inches='tight')
plt.close() # Dropped plt.show() memory leak pattern

# ---------- Figure 2: Weighted trends over time ----------
# Changed: Implemented dual axes (twinx). Corporate ownership variation was invisible 
# when forced to share a y-axis with the much larger owner-occupancy percentages.
def wavg(g, val, w):
    x = g[val].to_numpy()
    weights = g[w].to_numpy()
    m = np.isfinite(x) & np.isfinite(weights)
    if m.sum() == 0:
        return np.nan
    return np.average(x[m], weights=weights[m])

wtrend = (
    df.groupby("Year")
      .apply(lambda g: pd.Series({
          "corp_wavg": wavg(g, "corp_own_rate", "occ_unit"),
          "ownocc_wavg": wavg(g, "own_occ_rate", "occ_unit"),
      }))
      .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary axis
color1 = COLORS['primary']
ax1.plot(wtrend["Year"], wtrend["ownocc_wavg"], 
         marker='o', linewidth=2.5, markersize=7, 
         color=color1, label="Owner-Occupancy Rate")
ax1.set_xlabel("Year", fontweight='bold')
ax1.set_ylabel("Owner-Occupancy Rate", fontweight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Secondary axis
ax2 = ax1.twinx()
color2 = COLORS['quaternary']
ax2.plot(wtrend["Year"], wtrend["corp_wavg"], 
         marker='s', linewidth=2.5, markersize=7, 
         color=color2, label="Corporate Ownership Rate")
ax2.set_ylabel("Corporate Ownership Rate", fontweight='bold', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.title("Boston Housing Trends: Weighted Average Rates by Occupied Units", 
          fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax2.grid(False) # Turn off grid for secondary to avoid visual noise
sns.despine(ax=ax1, right=False)
sns.despine(ax=ax2, right=False)

# Combine legends from twin axes smoothly
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', frameon=True)

plt.tight_layout()
plt.savefig('fig2_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Figure 3: Top changes ----------
# Changed: Added explicit text annotations on the bars to quickly verify magnitudes
d2004 = df[df["Year"] == 2004].set_index("Neighborhood")
d2024 = df[df["Year"] == 2024].set_index("Neighborhood")

delta = pd.DataFrame({
    "delta_corp": d2024["corp_own_rate"] - d2004["corp_own_rate"],
    "delta_ownerocc": d2024["own_occ_rate"] - d2004["own_occ_rate"],
}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Top corporate ownership increases
top_inc = delta.sort_values("delta_corp", ascending=False).head(10)
y_pos1 = np.arange(len(top_inc))
axes[0].barh(y_pos1, top_inc["delta_corp"][::-1], color=COLORS['quaternary'], edgecolor='none')
axes[0].set_yticks(y_pos1)
axes[0].set_yticklabels(top_inc.index[::-1])
axes[0].set_title("Largest Increases in Corporate Ownership (2004-2024)", fontweight='bold')
axes[0].set_xlabel("Change in Rate")
axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
sns.despine(ax=axes[0])

# Bar annotations
for i, v in enumerate(top_inc["delta_corp"][::-1]):
    axes[0].text(v + 0.002, i, f"+{v*100:.1f}%", va='center', color='black', fontsize=10)

# Top owner-occupancy decreases
top_dec = delta.sort_values("delta_ownerocc", ascending=True).head(10)
y_pos2 = np.arange(len(top_dec))
axes[1].barh(y_pos2, top_dec["delta_ownerocc"][::-1], color=COLORS['primary'], edgecolor='none')
axes[1].set_yticks(y_pos2)
axes[1].set_yticklabels(top_dec.index[::-1])
axes[1].set_title("Largest Decreases in Owner-Occupancy (2004-2024)", fontweight='bold')
axes[1].set_xlabel("Change in Rate")
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
sns.despine(ax=axes[1])

# Bar annotations
for i, v in enumerate(top_dec["delta_ownerocc"][::-1]):
    axes[1].text(v - 0.002, i, f"{v*100:.1f}%", va='center', ha='right', color='black', fontsize=10)

# Dynamically extend xlim slightly to prevent text clipping
xmin, xmax = axes[0].get_xlim()
axes[0].set_xlim(xmin, xmax + 0.02)
xmin, xmax = axes[1].get_xlim()
axes[1].set_xlim(xmin - 0.02, xmax) 

plt.tight_layout()
plt.savefig('fig3_top_changes.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Figure 4: Scatter with demographics ----------
df2024 = df[df["Year"] == 2024].dropna(subset=["own_occ_rate", "corp_own_rate", "pct_nonwhite"]).copy()

fig, ax = plt.subplots(figsize=(12, 9))  # Slightly larger for better spacing

# Changed: Using YlOrRd colormap (yellow=low, orange=mid, red=high)
sc = ax.scatter(
    df2024["own_occ_rate"],
    df2024["corp_own_rate"],
    c=df2024["pct_nonwhite"],
    s=180,  # Slightly larger dots
    alpha=0.85,
    cmap='YlOrRd',
    edgecolors='white',
    linewidth=1.0
)

# Render linear trendline
m, b = np.polyfit(df2024["own_occ_rate"], df2024["corp_own_rate"], 1)
x_val = np.linspace(df2024["own_occ_rate"].min(), df2024["own_occ_rate"].max(), 100)
ax.plot(x_val, m*x_val + b, color='black', linestyle='--', alpha=0.6, linewidth=2, 
        label=f"Trendline (slope={m:.2f})")

# Changed: Smart labeling - only label the most notable neighborhoods
labeled_neighborhoods = set()

# Top 3 highest corporate ownership
top_corp = df2024.nlargest(3, "corp_own_rate")
for _, row in top_corp.iterrows():
    labeled_neighborhoods.add(row["Neighborhood"])
    ax.annotate(row["Neighborhood"], 
                (row["own_occ_rate"], row["corp_own_rate"]),
                xytext=(8, 8), textcoords="offset points", 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.95, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1, color='black'))

# Bottom 3 lowest owner-occupancy (if not already labeled)
low_ownocc = df2024.nsmallest(3, "own_occ_rate")
for _, row in low_ownocc.iterrows():
    if row["Neighborhood"] not in labeled_neighborhoods:
        labeled_neighborhoods.add(row["Neighborhood"])
        ax.annotate(row["Neighborhood"], 
                    (row["own_occ_rate"], row["corp_own_rate"]),
                    xytext=(8, -20), textcoords="offset points", 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.95, linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1, color='black'))

# Top 3 highest non-white population (if not already labeled)
top_diverse = df2024.nlargest(3, "pct_nonwhite")
for _, row in top_diverse.iterrows():
    if row["Neighborhood"] not in labeled_neighborhoods:
        labeled_neighborhoods.add(row["Neighborhood"])
        # Vary placement to reduce overlap
        offset_x = -80 if row["own_occ_rate"] > 0.4 else 8
        offset_y = 8
        ha = 'right' if offset_x < 0 else 'left'
        
        ax.annotate(row["Neighborhood"], 
                    (row["own_occ_rate"], row["corp_own_rate"]),
                    xytext=(offset_x, offset_y), textcoords="offset points", 
                    fontsize=9, fontweight='bold', ha=ha,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.95, linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1, color='black'))

ax.set_title("Corporate Ownership vs Owner-Occupancy by Neighborhood (2024)\nColored by Non-White Population Share", 
             fontweight='bold', pad=20, fontsize=14)
ax.set_xlabel("Owner-Occupancy Rate", fontweight='bold', fontsize=12)
ax.set_ylabel("Corporate Ownership Rate", fontweight='bold', fontsize=12)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

sns.despine(ax=ax)

cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Non-White Population Share", fontweight='bold', fontsize=11)
cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig('fig4_scatter_demographics.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Figure 5: Correlation stability ----------
# Changed: Implemented `fill_between` highlighting magnitude from the zero-line 
# visually separating positive and negative coefficients.
corr_by_year = (
    df.dropna(subset=["own_occ_rate", "corp_own_rate"])
      .groupby("Year")
      .apply(lambda g: g["corp_own_rate"].corr(g["own_occ_rate"]))
      .reset_index(name="correlation")
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(corr_by_year["Year"], corr_by_year["correlation"],
        marker='o', linewidth=2.5, markersize=8,
        color=COLORS['primary'])

# Shaded confidence/strength region
ax.fill_between(corr_by_year["Year"], 0, corr_by_year["correlation"], 
                color=COLORS['primary'], alpha=0.15)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

ax.set_title("Temporal Stability: Correlation Between Corporate Ownership and Owner-Occupancy", 
             fontweight='bold', pad=15)
ax.set_xlabel("Year", fontweight='bold')
ax.set_ylabel("Pearson Correlation Coefficient", fontweight='bold')
ax.set_ylim([-1, 1])

# Fix floating points in X-axis (plot every 2 years instead)
ax.set_xticks(corr_by_year["Year"][::2])

sns.despine()
plt.tight_layout()
plt.savefig('fig5_correlation_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("All figures saved successfully!")
print("=" * 60)


# ---------- Figure 6: Ownership archetypes (donut small multiples) ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Assumes df already loaded with: Year, Neighborhood, corp_own_rate, own_occ_rate
# Uses corp_own_rate (2024 "level") and delta corp_own_rate (2004->2024 "growth")

def make_archetype_donuts(df, outpath="fig6_archetypes_donuts.png"):
    # --- Build 2004/2024 comparison table ---
    d04 = df[df["Year"] == 2004][["Neighborhood", "corp_own_rate", "own_occ_rate"]].set_index("Neighborhood")
    d24 = df[df["Year"] == 2024][["Neighborhood", "corp_own_rate", "own_occ_rate"]].set_index("Neighborhood")

    dd = (
        d24.join(d04, lsuffix="_2024", rsuffix="_2004", how="inner")
          .dropna(subset=["corp_own_rate_2024", "corp_own_rate_2004", "own_occ_rate_2024"])
          .copy()
    )

    # --- Define "level" and "growth" clearly ---
    # Level: 2024 corporate ownership rate
    dd["level"] = dd["corp_own_rate_2024"]

    # Growth: change in corporate ownership from 2004 to 2024
    dd["growth"] = dd["corp_own_rate_2024"] - dd["corp_own_rate_2004"]

    # Median splits (robust, non-arbitrary)
    level_med = dd["level"].median()
    growth_med = dd["growth"].median()

    dd["level_bin"] = np.where(dd["level"] >= level_med, "High level", "Low level")
    dd["growth_bin"] = np.where(dd["growth"] >= growth_med, "High growth", "Low growth")

    # Quadrant labels in a consistent order
    quad_order = [
        ("High level", "High growth"),
        ("High level", "Low growth"),
        ("Low level", "High growth"),
        ("Low level", "Low growth"),
    ]

    # --- Aggregate ownership composition within each quadrant ---
    # Use 2024 composition in each quadrant:
    # corporate-owned = mean(corp_own_rate_2024)
    # owner-occupied = mean(own_occ_rate_2024)
    # other = remaining share (clip to [0,1] to avoid tiny negatives due to noise)
    quads = []
    for lb, gb in quad_order:
        sub = dd[(dd["level_bin"] == lb) & (dd["growth_bin"] == gb)].copy()
        n = len(sub)
        corp = float(sub["corp_own_rate_2024"].mean()) if n else np.nan
        own  = float(sub["own_occ_rate_2024"].mean()) if n else np.nan
        other = 1.0 - corp - own if n else np.nan

        if n:
            corp = float(np.clip(corp, 0, 1))
            own  = float(np.clip(own, 0, 1))
            other = float(np.clip(other, 0, 1))

        quads.append({
            "level_bin": lb,
            "growth_bin": gb,
            "n": n,
            "corp": corp,
            "own": own,
            "other": other
        })

    quad_df = pd.DataFrame(quads)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Donut settings
    labels = ["Corporate-owned", "Owner-occupied", "Other (e.g., small landlords, mixed)"]

    # Keep colors simple; if you want to reuse your COLORS dict, swap these out.
    # (Not specifying colors is fine too, but consistent legend colors help readability.)
    donut_colors = ["#B94A2C", "#3E7FA5", "#6C757D"]

    for ax, row in zip(axes, quad_df.itertuples(index=False)):
        if row.n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        sizes = [row.corp, row.own, row.other]

        ax.pie(
            sizes,
            startangle=90,
            colors=donut_colors,
            wedgeprops=dict(width=0.35, edgecolor="white")
        )
        ax.set(aspect="equal")

        # Title includes quadrant + n-count
        ax.set_title(f"{row.level_bin} / {row.growth_bin}\n(n = {row.n})", fontweight="bold", fontsize=12)

    # Global title + definition subtitle
    fig.suptitle(
        "Neighborhood Ownership Archetypes\n2024 Corporate Ownership Level vs 2004–2024 Corporate Ownership Growth",
        fontsize=16, fontweight="bold", y=0.97
    )
    fig.text(
        0.5, 0.92,
        f"Level = corp_own_rate (2024), Growth = Δcorp_own_rate (2024 − 2004); splits are citywide medians "
        f"(level median={level_med:.1%}, growth median={growth_med:.1%}).",
        ha="center", fontsize=10
    )

    # Legend
    fig.legend(
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04)
    )

    # --- Overlay conceptual X/Y axis arrows between donuts (figure coordinates) ---
    # These are directional cues, not numeric axes.
    # Center point between the 4 subplots in figure fraction coords:
    cx, cy = 0.5, 0.52

    # X arrow: Level increases to the right
    x_arrow = FancyArrowPatch(
        (cx - 0.10, cy), (cx + 0.10, cy),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
        lw=1.5, color="black", alpha=0.8
    )

    # Y arrow: Growth increases upward
    y_arrow = FancyArrowPatch(
        (cx, cy - 0.10), (cx, cy + 0.10),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
        lw=1.5, color="black", alpha=0.8
    )

    fig.add_artist(x_arrow)
    fig.add_artist(y_arrow)

    fig.text(cx + 0.11, cy, "Higher 2024 corporate ownership (Level)", transform=fig.transFigure,
             ha="left", va="center", fontsize=10)
    fig.text(cx, cy + 0.11, "Higher 2004→2024 increase (Growth)", transform=fig.transFigure,
             ha="center", va="bottom", fontsize=10, rotation=90)

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.90])
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

# Call it after your other figures:
make_archetype_donuts(df, outpath="fig6_archetypes_donuts.png")
