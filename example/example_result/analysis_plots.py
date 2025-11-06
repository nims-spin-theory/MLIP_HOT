#!/usr/bin/env python3
"""
Analysis and Visualization Script for DFT vs ML Comparison
This script compares DFT and ML predictions for cell parameters, 
formation energy, and hull distance.
"""

import pandas as pd
import numpy as np
import ast
from typing import Optional
import matplotlib.pyplot as plt


def str_to_2d_array(string: str) -> Optional[np.ndarray]:
    """
    Convert a string representation of a 2D array to a NumPy array.
    
    Args:
        string: String representation of a 2D array
        
    Returns:
        NumPy array if conversion is successful, None otherwise
    """
    if ',' not in string:
        string = string.replace(' ', ',')
    try:
        list_of_lists = ast.literal_eval(string)
        return np.array(list_of_lists)
    except (ValueError, SyntaxError):
        return None


def create_comparison_plot(dft_data, ml_data, xlabel, ylabel, title, unit):
    """
    Create a comparison plot between DFT and ML data with statistics.
    
    Args:
        dft_data: DFT reference data
        ml_data: ML predicted data
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        unit: Unit string for statistics display
    """
    # Create improved plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(dft_data, ml_data, 'o', markersize=8, alpha=0.6, label='Data points')
    
    # Add perfect correlation line
    min_val = min(min(dft_data), min(ml_data))
    max_val = max(max(dft_data), max(ml_data))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect correlation')
    
    # Calculate statistics
    dft_arr = np.array(dft_data)
    ml_arr = np.array(ml_data)
    correlation = np.corrcoef(dft_arr, ml_arr)[0, 1]
    r_squared = correlation ** 2
    mae = np.mean(np.abs(dft_arr - ml_arr))
    rmse = np.sqrt(np.mean((dft_arr - ml_arr) ** 2))
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add statistics text box
    stats_text = f'R² = {r_squared:.4f}\nMAE = {mae:.4f} {unit}\nRMSE = {rmse:.4f} {unit}\nN = {len(dft_data)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax


def plot_cell_parameters(df_dft, df_ml):
    """Plot comparison of cell parameters."""
    print("Plotting cell parameter comparison...")
    
    # Extract cell[0,0] values
    dft = []
    ml = []
    for ind, row in df_dft.iterrows():
        dft.append(str_to_2d_array(row['cell'])[0][0])
    for ind, row in df_ml.iterrows():
        ml.append(str_to_2d_array(row['optimized_cell'])[0][0])
    
    fig, ax = create_comparison_plot(
        dft, ml,
        xlabel='DFT Cell[0,0] (Å)',
        ylabel='ML Cell[0,0] (Å)',
        title='DFT vs ML Cell Parameters (MatterSim)',
        unit='Å'
    )
    plt.savefig('cell_parameters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Cell parameter plot saved as 'cell_parameters_comparison.png'")


def plot_formation_energy(df_dft, df_ml):
    """Plot comparison of formation energy."""
    print("Plotting formation energy comparison...")
    
    dft = df_dft['formation energy (eV/atom)']
    ml = df_ml['Formation Energy (eV/atom)']
    
    fig, ax = create_comparison_plot(
        dft, ml,
        xlabel='DFT Formation Energy (eV/atom)',
        ylabel='ML Formation Energy (eV/atom)',
        title='DFT vs ML Formation Energy (MatterSim)',
        unit='eV/atom'
    )
    plt.savefig('formation_energy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Formation energy plot saved as 'formation_energy_comparison.png'")


def plot_hull_distance(df_dft, df_ml):
    """Plot comparison of hull distance."""
    print("Plotting hull distance comparison...")
    
    dft = df_dft['hull distance (eV/atom)']
    ml = df_ml['Hull Distance (eV/atom)']
    
    fig, ax = create_comparison_plot(
        dft, ml,
        xlabel='DFT Hull Distance (eV/atom)',
        ylabel='ML Hull Distance (eV/atom)',
        title='DFT vs ML Hull Distance (MatterSim)',
        unit='eV/atom'
    )
    plt.savefig('hull_distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Hull distance plot saved as 'hull_distance_comparison.png'")


def main():
    """Main function to run all analyses."""
    print("="*60)
    print("DFT vs ML Comparison Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df_dft = pd.read_csv('./example_data_dft.csv', index_col=0)
    df_ml = pd.read_csv('./example_data_result_oqmd.csv', index_col=0)
    
    # Sort both dataframes by 'composition' to ensure alignment
    df_dft = df_dft.sort_values('composition')
    df_ml = df_ml.sort_values('composition')
    
    print(f"Loaded {len(df_dft)} DFT structures")
    print(f"Loaded {len(df_ml)} ML structures")
    
    # Generate all plots
    print("\n" + "="*60)
    plot_cell_parameters(df_dft, df_ml)
    
    print("\n" + "="*60)
    plot_formation_energy(df_dft, df_ml)
    
    print("\n" + "="*60)
    plot_hull_distance(df_dft, df_ml)
    
    print("\n" + "="*60)
    print("Analysis complete! All plots have been generated.")
    print("="*60)


if __name__ == "__main__":
    main()
