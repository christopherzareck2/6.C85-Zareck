import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load ---
df = pd.read_csv("Census_and_Corp_Ownership_and_Occupancy_Over_Time.csv")

# --- Basic structure checks ---
print("Rows, cols:", df.shape)
print("Years:", df["Year"].min(), "to", df["Year"].max())
print("Neighborhoods:", df["Neighborhood"].nunique())
print("Duplicate Neighborhood-Year rows:", df.duplicated(["Neighborhood", "Year"]).sum())
print("Total missing cells:", df.isna().sum().sum())

# Rates should be in [0, 1]
for col in ["corp_own_rate", "own_occ_rate"]:
    print(col, "min/max:", df[col].min(), df[col].max())

# --- Output directory for images ---
outdir = "a2_figs"
os.makedirs(outdir, exist_ok=True)

# Pick a snapshot year for "neighborhood comparisons"
END_YEAR = int(df["Year"].max())
snap = df[df["Year"] == END_YEAR].copy()

# --- Derive demographic shares (for correlation-style plots) ---
# NOTE: these are counts; shares are often easier to compare across neighborhoods.
snap["pct_black_all"] = snap["black_all"] / snap["tot_pop_all"]
snap["pct_hisp_all"]  = snap["hisp_all"]  / snap["tot_pop_all"]

# --- Phase 1 visuals: distributions (2024) ---
plt.figure()
plt.hist(snap["corp_own_rate"], bins=10)
plt.xlabel(f"Corporate ownership rate ({END_YEAR})")
plt.ylabel("Number of neighborhoods")
plt.title(f"Distribution of corporate ownership rate ({END_YEAR})")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "hist_corp.png"), dpi=200)
plt.close()

plt.figure()
plt.hist(snap["own_occ_rate"], bins=10)
plt.xlabel(f"Owner-occupancy rate ({END_YEAR})")
plt.ylabel("Number of neighborhoods")
plt.title(f"Distribution of owner-occupancy rate ({END_YEAR})")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "hist_ownerocc.png"), dpi=200)
plt.close()

# --- Q1: trends over time (weighted by occupied units) ---
df["weight"] = df["occ_unit"]
agg = df.groupby("Year").apply(lambda g: pd.Series({
    "corp_wavg": np.average(g["corp_own_rate"], weights=g["weight"]),
    "ownocc_wavg": np.average(g["own_occ_rate"], weights=g["weight"]),
})).reset_index()

plt.figure()
plt.plot(agg["Year"], agg["corp_wavg"], label="Corporate ownership (weighted avg)")
plt.plot(agg["Year"], agg["ownocc_wavg"], label="Owner-occupancy (weighted avg)")
plt.ylim(0, 1)
plt.xlabel("Year")
plt.ylabel("Rate")
plt.title("Boston trends (weighted by occupied units)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "trend_weighted.png"), dpi=200)
plt.close()

# --- Q2: neighborhood change from first to last year ---
START_YEAR = int(df["Year"].min())
base = df[df["Year"] == START_YEAR].set_index("Neighborhood")
end  = df[df["Year"] == END_YEAR].set_index("Neighborhood")
common = base.index.intersection(end.index)

chg_corp = (end.loc[common, "corp_own_rate"] - base.loc[common, "corp_own_rate"]).sort_values()
chg_own  = (end.loc[common, "own_occ_rate"]  - base.loc[common, "own_occ_rate"]).sort_values()

plt.figure(figsize=(8, 5))
plt.barh(chg_corp.index, chg_corp.values)
plt.xlabel(f"Change in corporate ownership rate ({END_YEAR} - {START_YEAR})")
plt.title(f"Corporate ownership change by neighborhood, {START_YEAR}→{END_YEAR}")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "chg_corp_barh.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(chg_own.index, chg_own.values)
plt.xlabel(f"Change in owner-occupancy rate ({END_YEAR} - {START_YEAR})")
plt.title(f"Owner-occupancy change by neighborhood, {START_YEAR}→{END_YEAR}")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "chg_ownerocc_barh.png"), dpi=200)
plt.close()

# --- Q3: relationships (2024) ---
plt.figure()
plt.scatter(snap["own_occ_rate"], snap["corp_own_rate"], s=20)
plt.xlabel(f"Owner-occupancy rate ({END_YEAR})")
plt.ylabel(f"Corporate ownership rate ({END_YEAR})")
plt.title(f"Corporate ownership vs owner-occupancy ({END_YEAR})")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "scatter_corp_vs_ownerocc.png"), dpi=200)
plt.close()

plt.figure()
plt.scatter(snap["pct_black_all"], snap["corp_own_rate"], s=20)
plt.xlabel("Black population share (all ages; snapshot)")
plt.ylabel(f"Corporate ownership rate ({END_YEAR})")
plt.title("Corporate ownership vs Black population share")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "scatter_corp_vs_pct_black.png"), dpi=200)
plt.close()

plt.figure()
plt.scatter(snap["pct_hisp_all"], snap["corp_own_rate"], s=20)
plt.xlabel("Hispanic/Latinx population share (all ages; snapshot)")
plt.ylabel(f"Corporate ownership rate ({END_YEAR})")
plt.title("Corporate ownership vs Hispanic/Latinx population share")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "scatter_corp_vs_pct_hisp.png"), dpi=200)
plt.close()

print("Done. Figures saved to:", outdir)