"""
COVID-19 Spiral Heatmap Visualization

Creates a polar coordinate heatmap showing monthly COVID-19 case patterns
in the United States from 2020-2022, with years as concentric rings.

Note: Uses logarithmic color scale to better visualize the wide range of values
(Omicron peak ~660k vs typical months <150k).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Constants
DATA_FILE = "COVID_US_cases.csv"
ROLLING_WINDOW = 7
TARGET_YEARS = [2020, 2021, 2022]
MONTHS_IN_YEAR = 12
SCALE_TO_THOUSANDS = 1000.0

# Visualization constants
FIGURE_SIZE = (10.5, 10.5)
DPI = 180
COLOR_PALETTE = ["#f7f7f7", "#fdd0d0", "#fb6a6a", "#cb181d"]
MISSING_VALUE_COLOR = "#e6e6e6"


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load COVID-19 case data and calculate 7-day rolling averages.
    
    Args:
        file_path: Path to the COVID-19 CSV data file
        
    Returns:
        DataFrame with date, year, month, and 7-day rolling average columns
    """
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()
    
    # Calculate 7-day rolling average (matches NYT style)
    df["cases_7day_avg"] = (
        df["new_confirmed"]
        .clip(lower=0)  # Avoid negative backfill corrections
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )
    
    # Filter to target years and extract time components
    df = df[df["date"].dt.year.isin(TARGET_YEARS)].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    
    return df


def aggregate_to_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to monthly means and reshape to matrix format.
    
    Args:
        df: DataFrame with year, month, and cases_7day_avg columns
        
    Returns:
        Pivot table with years as rows, months as columns, values in thousands
    """
    # Aggregate to monthly mean of 7-day average
    monthly = (
        df.groupby(["year", "month"], as_index=False)["cases_7day_avg"]
        .mean()
    )
    
    # Pivot to matrix [year x month]
    heat = monthly.pivot(index="year", columns="month", values="cases_7day_avg")
    
    # Ensure all months and years exist (fill gaps with NaN)
    heat = heat.reindex(index=TARGET_YEARS, columns=range(1, MONTHS_IN_YEAR + 1))
    
    # Convert to thousands for better readability
    return heat / SCALE_TO_THOUSANDS



def create_custom_colormap():
    """
    Create a custom red color palette for COVID-19 heatmap.
    
    Returns:
        LinearSegmentedColormap configured for the visualization
    """
    cmap = LinearSegmentedColormap.from_list(
        "covid_reds",
        COLOR_PALETTE
    ).copy()
    cmap.set_bad(MISSING_VALUE_COLOR)  # Light gray for missing values
    return cmap


def setup_polar_axes(ax):
    """
    Configure polar axes for spiral heatmap layout.
    
    Args:
        ax: Matplotlib polar axes object
    """
    # Polar orientation to resemble calendar layout
    ax.set_theta_zero_location("N")  # January at top
    ax.set_theta_direction(-1)  # Clockwise months
    ax.set_yticks([])
    ax.set_ylim(0.9, 4.35)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


def draw_heatmap_tiles(ax, data: np.ndarray, cmap, use_log_scale: bool = True):
    """
    Draw the heatmap tiles in polar coordinates.
    
    Args:
        ax: Matplotlib polar axes object
        data: 2D array of case counts (years x months)
        cmap: Colormap to use
        use_log_scale: Whether to use logarithmic color scale (recommended for wide value ranges)
        
    Returns:
        Tuple of (mesh, theta_edges, r_edges, vmax, vmin)
    """
    masked_data = np.ma.masked_invalid(data)
    
    # Angular edges for 12 months (0 to 2π)
    theta_edges = np.linspace(0, 2 * np.pi, MONTHS_IN_YEAR + 1)
    
    # Radial edges for 3 year rings (inner to outer: 2020, 2021, 2022)
    r_edges = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Create meshgrid for pcolormesh
    Theta, R = np.meshgrid(theta_edges, r_edges)
    
    # Calculate min/max, avoiding zeros for log scale
    vmax = np.nanmax(data)
    vmin = np.nanmin(data[data > 0]) if use_log_scale else 0
    
    # Draw heatmap tiles with logarithmic or linear scale
    if use_log_scale:
        # Use log scale to better show variation across wide range
        mesh = ax.pcolormesh(
            Theta,
            R,
            masked_data,
            cmap=cmap,
            shading="flat",
            norm=LogNorm(vmin=max(vmin, 0.1), vmax=vmax),  # Avoid log(0)
            edgecolors="white",
            linewidth=1.5
        )
    else:
        # Linear scale (original approach)
        mesh = ax.pcolormesh(
            Theta,
            R,
            masked_data,
            cmap=cmap,
            shading="flat",
            vmin=0,
            vmax=vmax,
            edgecolors="white",
            linewidth=1.5
        )
    
    return mesh, theta_edges, r_edges, vmax, vmin


def add_month_labels(ax, theta_edges):
    """
    Add month labels around the perimeter.
    
    Args:
        ax: Matplotlib polar axes object
        theta_edges: Array of angular positions for month boundaries
    """
    month_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_centers)
    ax.set_xticklabels(month_labels, fontsize=10)
    return month_centers


def add_guide_lines(ax, theta_edges):
    """
    Add circular guide lines and radial spokes for readability.
    
    Args:
        ax: Matplotlib polar axes object
        theta_edges: Array of angular positions for month boundaries
    """
    # Circular guide lines between years
    for r in [1, 2, 3, 4]:
        ax.plot(np.linspace(0, 2 * np.pi, 300), np.full(300, r),
                color="white", lw=1.2, alpha=0.9)
    
    # Light month spokes
    for t in theta_edges[:-1]:
        ax.plot([t, t], [1.0, 4.0], color="white", lw=1.0, alpha=0.7)


def add_year_labels(ax):
    """
    Add year labels to each concentric ring.
    
    Args:
        ax: Matplotlib polar axes object
    """
    year_r_centers = [1.5, 2.5, 3.5]
    year_names = ["2020", "2021", "2022"]
    label_angle = np.deg2rad(35)  # Place labels around 1 o'clock
    
    for r, year in zip(year_r_centers, year_names):
        ax.text(label_angle, r, year, fontsize=11, fontweight="bold",
                ha="left", va="center", color="black")


def add_cell_annotations(ax, heat_data: pd.DataFrame, month_centers, r_edges, vmax, use_log_scale: bool = True):
    """
    Add numerical values to each cell in the heatmap.
    
    Args:
        ax: Matplotlib polar axes object
        heat_data: DataFrame with case counts in thousands
        month_centers: Array of angular positions for month centers
        r_edges: Array of radial positions for year boundaries
        vmax: Maximum value for determining text color
        use_log_scale: Whether log scale is used (affects color threshold)
    """
    for i, year in enumerate(TARGET_YEARS):
        for j, month in enumerate(range(1, MONTHS_IN_YEAR + 1)):
            val = heat_data.loc[year, month]
            theta_c = month_centers[j]
            r_c = (r_edges[i] + r_edges[i + 1]) / 2
            
            if pd.isna(val):
                ax.text(theta_c, r_c, "NA", ha="center", va="center",
                        fontsize=8, color="dimgray")
                continue
            
            # Text color based on intensity
            # For log scale, use a different threshold since colors are distributed differently
            if use_log_scale:
                text_color = "white" if val > 100 else "black"
            else:
                text_color = "white" if val > (vmax * 0.40) else "black"
            
            ax.text(theta_c, r_c, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color=text_color)


def add_omicron_annotation(ax, month_centers):
    """
    Add annotation highlighting the Omicron peak.
    
    Args:
        ax: Matplotlib polar axes object
        month_centers: Array of angular positions for month centers
    """
    # January 2022 is in outer ring, first month
    jan_idx = 0
    theta_jan = month_centers[jan_idx]
    r_omicron = 3.5
    
    ax.annotate(
        "Omicron peak (Jan 2022)",
        xy=(theta_jan, r_omicron),
        xytext=(np.deg2rad(300), 4.25),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=10,
        ha="center",
        va="center"
    )


def add_titles_and_colorbar(fig, ax, mesh, vmax, use_log_scale: bool = True):
    """
    Add title, subtitle, colorbar, and source information.
    
    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        mesh: The mesh object from pcolormesh
        vmax: Maximum value for colorbar ticks
        use_log_scale: Whether logarithmic scale is used
    """
    # Main title
    fig.suptitle(
        "U.S. COVID-19 Cases by Month (2020–2022)",
        fontsize=15,
        y=0.98
    )
    
    # Subtitle with note about scale
    scale_note = " (logarithmic scale)" if use_log_scale else ""
    fig.text(
        0.5, 0.945,
        f"Monthly mean of 7-day average new confirmed cases (thousands){scale_note}",
        ha="center", va="center", fontsize=10.5
    )
    
    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax, pad=0.10, shrink=0.78)
    cbar.set_label("Monthly mean of 7-day avg new cases (thousands)", fontsize=10)
    
    if use_log_scale:
        # For log scale, let matplotlib handle the tick placement
        cbar.ax.minorticks_off()
    else:
        # For linear scale, use nice round ticks
        tick_max = int(np.ceil(vmax / 100.0) * 100)
        if vmax > tick_max * 0.95:
            tick_max += 100
        cbar.set_ticks(np.arange(0, tick_max + 1, 100))
    
    # Source note
    fig.text(
        0.02, 0.02,
        "Source: Google COVID-19 Open Data | Transform: 7-day rolling average, "
        "aggregated to monthly means | Gray cells indicate missing months",
        ha="left", va="bottom", fontsize=8, color="dimgray"
    )


def create_spiral_heatmap(heat_data: pd.DataFrame, output_file: str = "a3_final_spiral_heatmap.png",
                          use_log_scale: bool = True):
    """
    Create and save the spiral heatmap visualization.
    
    Args:
        heat_data: DataFrame with monthly case counts in thousands
        output_file: Path to save the output image
        use_log_scale: Whether to use logarithmic color scale (recommended for wide value ranges)
    """
    # Create figure with polar axes
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = plt.subplot(111, projection="polar")
    
    # Setup and draw
    setup_polar_axes(ax)
    cmap = create_custom_colormap()
    data = heat_data.values
    mesh, theta_edges, r_edges, vmax, vmin = draw_heatmap_tiles(ax, data, cmap, use_log_scale)
    
    # Add labels and annotations
    month_centers = add_month_labels(ax, theta_edges)
    add_guide_lines(ax, theta_edges)
    add_year_labels(ax)
    add_cell_annotations(ax, heat_data, month_centers, r_edges, vmax, use_log_scale)
    add_omicron_annotation(ax, month_centers)
    
    # Add titles and colorbar
    add_titles_and_colorbar(fig, ax, mesh, vmax, use_log_scale)
    
    # Final layout and save
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    # Load and process data
    df = load_and_preprocess_data(DATA_FILE)
    heat_data = aggregate_to_monthly_data(df)
    
    # Print data summary for understanding
    print("=" * 60)
    print("COVID-19 Data Summary (values in thousands)")
    print("=" * 60)
    print(f"Date range: 2020-2022")
    print(f"Min value: {heat_data.min().min():.1f}k")
    print(f"Max value: {heat_data.max().max():.1f}k (Omicron peak, Jan 2022)")
    print(f"Median: {heat_data.median().median():.1f}k")
    print(f"Mean: {heat_data.mean().mean():.1f}k")
    print(f"\nN/A values: {heat_data.isna().sum().sum()} months")
    if heat_data.isna().sum().sum() > 0:
        print("Note: N/A values indicate months with missing or incomplete data")
    print(f"\nUsing logarithmic scale to better visualize the wide range")
    print(f"(Omicron was ~4.5x higher than most other peaks)")
    print("=" * 60)
    print()
    
    # Create visualization with log scale (default)
    create_spiral_heatmap(heat_data, use_log_scale=True)



if __name__ == "__main__":
    main()
