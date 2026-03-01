"""
A4 - CCRB Complaint Data Analysis
Import and explore CCRB (Civilian Complaint Review Board) complaint data

Proposition: A relatively small group of officers accounts for a 
disproportionate share of civilian misconduct allegations.
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEGATIONS_FILE = os.path.join(BASE_DIR, "allegations_202007271729.csv")
LAYOUT_FILE = os.path.join(BASE_DIR, "CCRB Data Layout Table.xlsx")

def load_allegations_data():
    """
    Load the allegations CSV file
    Returns: DataFrame with allegations data
    """
    print("Loading allegations data...")
    df = pd.read_csv(ALLEGATIONS_FILE)
    print(f"Loaded {len(df):,} allegations records")
    print(f"Columns: {list(df.columns)}")
    return df

def load_layout_table():
    """
    Load the CCRB Data Layout Table Excel file
    Returns: Dictionary of DataFrames (one per sheet)
    """
    print("\nLoading data layout table...")
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(LAYOUT_FILE)
    print(f"Available sheets: {excel_file.sheet_names}")
    
    # Load each sheet into a dictionary
    layout_dict = {}
    for sheet_name in excel_file.sheet_names:
        layout_dict[sheet_name] = pd.read_excel(LAYOUT_FILE, sheet_name=sheet_name)
        print(f"  - Loaded sheet '{sheet_name}' with {len(layout_dict[sheet_name])} rows")
    
    return layout_dict

def explore_data(df):
    """
    Print basic information about the allegations data
    """
    print("\n" + "="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def create_for_visualization(df):
    """
    FOR SIDE: Clean Lorenz/Concentration Curve
    Shows that a small subset of officers accounts for a large share of allegations
    Uses allegation rows per officer (strongest concentration measure)
    """
    print("\n" + "="*80)
    print("Creating FOR visualization: Lorenz Concentration Curve")
    print("="*80)
    
    # Count allegations per officer
    officer_allegations = df.groupby('unique_mos_id').size().reset_index(name='allegation_count')
    officer_allegations = officer_allegations.sort_values('allegation_count', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentages
    total_officers = len(officer_allegations)
    total_allegations = officer_allegations['allegation_count'].sum()
    
    officer_allegations['cumulative_officers_pct'] = (np.arange(1, total_officers + 1) / total_officers) * 100
    officer_allegations['cumulative_allegations'] = officer_allegations['allegation_count'].cumsum()
    officer_allegations['cumulative_allegations_pct'] = (officer_allegations['cumulative_allegations'] / total_allegations) * 100
    
    # Calculate key statistics
    top_10_pct_cutoff = int(total_officers * 0.10)
    top_10_allegations_pct = officer_allegations.iloc[top_10_pct_cutoff - 1]['cumulative_allegations_pct']
    top_20_pct_cutoff = int(total_officers * 0.20)
    top_20_allegations_pct = officer_allegations.iloc[top_20_pct_cutoff - 1]['cumulative_allegations_pct']
    
    print(f"Top 10% of officers account for {top_10_allegations_pct:.1f}% of allegations")
    print(f"Top 20% of officers account for {top_20_allegations_pct:.1f}% of allegations")
    
    # Create the figure
    fig = go.Figure()
    
    # Add the perfect equality line (45-degree line)
    fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Perfect Equality',
        line=dict(color='#999999', width=2.5, dash='dash'),
        hovertemplate='<b>Perfect equality</b><br>%{x:.0f}% of officers → %{y:.0f}% of allegations<extra></extra>',
        showlegend=True
    ))
    
    # Add the actual concentration curve with area fill
    x_data = [0] + officer_allegations['cumulative_officers_pct'].tolist()
    y_data = [0] + officer_allegations['cumulative_allegations_pct'].tolist()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Actual Distribution',
        line=dict(color='#c7254e', width=4),
        fill='tonexty',
        fillcolor='rgba(199, 37, 78, 0.12)',
        hovertemplate='<b>Actual distribution</b><br>%{x:.1f}% of officers<br>%{y:.1f}% of allegations<extra></extra>',
        showlegend=True
    ))
    
    # Add a subtle reference line at 10% mark
    fig.add_shape(
        type="line",
        x0=10, y0=0, x1=10, y1=top_10_allegations_pct,
        line=dict(color='#c7254e', width=1.5, dash='dot'),
        opacity=0.4
    )
    
    fig.add_shape(
        type="line",
        x0=0, y0=top_10_allegations_pct, x1=10, y1=top_10_allegations_pct,
        line=dict(color='#c7254e', width=1.5, dash='dot'),
        opacity=0.4
    )
    
    # Update layout with cleaner styling
    fig.update_layout(
        title=dict(
            text="<b>A Small Fraction of Officers Account for a Large Share of Misconduct Allegations</b><br>" +
                 "<span style='font-size:14px; color:#666666'>Cumulative distribution of CCRB allegations across NYPD officers (1985–2020)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
            y=0.97,
            yanchor='top'
        ),
        xaxis=dict(
            title="<b>Cumulative Share of Officers (%)</b>",
            range=[-2, 102],
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.08)',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, family="Inter, sans-serif"),
            tickfont=dict(size=12),
            showline=True,
            linewidth=1,
            linecolor='#d0d0d0'
        ),
        yaxis=dict(
            title="<b>Cumulative Share of Allegations (%)</b>",
            range=[-2, 102],
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.08)',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, family="Inter, sans-serif"),
            tickfont=dict(size=12),
            showline=True,
            linewidth=1,
            linecolor='#d0d0d0'
        ),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        width=900,
        height=700,
        showlegend=True,
        legend=dict(
            x=0.03,
            y=0.97,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#d0d0d0',
            borderwidth=1,
            font=dict(size=12, family="Inter, sans-serif")
        ),
        hovermode='closest',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig, officer_allegations


def create_against_visualization(df):
    """
    AGAINST SIDE: Histogram of Unique Complaints per Officer
    Shows that complaint histories are spread across thousands of officers
    Uses unique complaints per officer (softens concentration)
    """
    print("\n" + "="*80)
    print("Creating AGAINST visualization: Distribution Histogram")
    print("="*80)
    
    # Count unique complaints per officer
    officer_complaints = df.groupby('unique_mos_id')['complaint_id'].nunique().reset_index(name='complaint_count')
    
    # Create bins
    bins = [0, 1, 3, 5, 10, float('inf')]
    labels = ['1', '2–3', '4–5', '6–10', '11+']
    officer_complaints['bin'] = pd.cut(officer_complaints['complaint_count'], 
                                       bins=bins, 
                                       labels=labels, 
                                       include_lowest=True)
    
    # Count officers in each bin
    bin_counts = officer_complaints['bin'].value_counts().sort_index()
    total_officers = len(officer_complaints)
    
    print("\nDistribution of officers by complaint count:")
    for bin_label, count in bin_counts.items():
        pct = (count / total_officers) * 100
        print(f"  {bin_label} complaints: {count:,} officers ({pct:.1f}%)")
    
    # Create the figure with a gradient from dark to light green (emphasizing the low-count majority)
    colors = ['#2d6a4f', '#40916c', '#52b788', '#74c69d', '#95d5b2']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_counts.index.tolist(),
        y=bin_counts.values,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2.5)
        ),
        text=[f"<b>{val:,}</b><br>{val/total_officers*100:.1f}%" for val in bin_counts.values],
        textposition='outside',
        textfont=dict(size=14, family="Inter, sans-serif", color='#333333'),
        hovertemplate='<b>%{x} complaints</b><br>' +
                      '%{y:,} officers<br>' +
                      '%{customdata:.1f}% of all officers<extra></extra>',
        customdata=[val/total_officers*100 for val in bin_counts.values]
    ))
    
    # Update layout with cleaner styling
    fig.update_layout(
        title=dict(
            text="<b>Complaint Histories Were Spread Across Thousands of Officers</b><br>" +
                 "<span style='font-size:14px; color:#666666'>Distribution of NYPD officers by unique complaint count (1985–2020)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
            y=0.97,
            yanchor='top'
        ),
        xaxis=dict(
            title="<b>Number of Unique Complaints per Officer</b>",
            showgrid=False,
            zeroline=False,
            title_font=dict(size=14, family="Inter, sans-serif"),
            tickfont=dict(size=13),
            showline=True,
            linewidth=1,
            linecolor='#d0d0d0'
        ),
        yaxis=dict(
            title="<b>Number of Officers</b>",
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.08)',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, family="Inter, sans-serif"),
            tickfont=dict(size=12),
            showline=True,
            linewidth=1,
            linecolor='#d0d0d0',
            range=[0, max(bin_counts.values) * 1.15]  # Add 15% padding for labels
        ),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        width=900,
        height=700,
        showlegend=False,
        hovermode='closest',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig, officer_complaints


def save_visualizations(for_fig, against_fig):
    """
    Save both visualizations as static PNG files only
    """
    print("\n" + "="*80)
    print("Saving visualizations...")
    print("="*80)
    
    # Save as PNG (static)
    for_fig.write_image(os.path.join(BASE_DIR, "fig_for_concentration.png"), 
                        width=900, height=700, scale=2)
    against_fig.write_image(os.path.join(BASE_DIR, "fig_against_distribution.png"), 
                           width=900, height=700, scale=2)
    print("✓ Saved PNG files")
    
    print("\nFiles saved:")
    print("  - fig_for_concentration.png")
    print("  - fig_against_distribution.png")


def main():
    """
    Main function to load and explore the CCRB data
    """
    print("="*80)
    print("CCRB COMPLAINT DATA IMPORT & ANALYSIS")
    print("="*80)
    
    # Load the allegations data
    allegations_df = load_allegations_data()
    
    # Load the layout table
    layout_dict = load_layout_table()
    
    # Explore the allegations data
    allegations_df = explore_data(allegations_df)
    
    # Create visualizations
    for_fig, officer_allegations = create_for_visualization(allegations_df)
    against_fig, officer_complaints = create_against_visualization(allegations_df)
    
    # Save visualizations
    save_visualizations(for_fig, against_fig)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nYou can now work with:")
    print("  - allegations_df: Main allegations DataFrame")
    print("  - layout_dict: Dictionary containing layout information sheets")
    print("  - for_fig: FOR side visualization (concentration curve)")
    print("  - against_fig: AGAINST side visualization (distribution histogram)")
    
    return allegations_df, layout_dict, for_fig, against_fig

if __name__ == "__main__":
    allegations_df, layout_dict, for_fig, against_fig = main()
