"""
Generate time-series curves for petroleum-based homopolymers for QA purposes.
Uses the existing model from modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.core_model import (
    sus, DAYS, SAMPLE_AREA, THICKNESS_DEFAULT,
    generate_material_curve, parse_thickness, find_material_by_grade,
    is_home_compostable_certified
)

def filter_petroleum_materials(sus_data):
    """Filter sustainability data to include both petroleum-based and PLA materials"""
    low_disintegration = ['LDPE', 'PP', 'PET', 'PVDC', 'PA', 'EVOH', 'Bio-PE', 'PLA']
    filtered_sus = sus_data[sus_data['Polymer Category'].isin(low_disintegration)].copy()
    return filtered_sus

def generate_homopolymer_curves():
    """Generate time-series curves for all low-disintegration homopolymers (petroleum + PLA)"""
    
    # Filter for low-disintegration materials (petroleum + PLA)
    low_disintegration_sus = filter_petroleum_materials(sus)
    print(f"Found {len(low_disintegration_sus)} low-disintegration materials (petroleum + PLA)")
    
    # Create figure for all curves - ZOOMED IN (0-5%)
    plt.figure(figsize=(15, 10))
    
    # Get a color map for unique colors
    cmap = plt.cm.get_cmap('tab20')
    
    for idx, (_, row) in enumerate(low_disintegration_sus.iterrows()):
        grade = row['Grade']
        polymer = row['Polymer Category']
        tuv_home = str(row.get('TUV Home', ''))
        thickness = parse_thickness(row.get('Thickness 1', None))
        
        # Use a unique seed for each material
        material_seed = hash(f"{polymer}_{grade}") % (2**32)
        
        # Generate curve using existing model
        curve = generate_material_curve(
            polymer, grade, tuv_home, thickness, material_seed=material_seed
        )
        
        # Plot curve with unique color
        days = range(0, DAYS)  # Start at 0 instead of 1
        is_home = is_home_compostable_certified(tuv_home)
        color = cmap(idx / len(low_disintegration_sus))  # Unique color for each curve
        linestyle = '-'  # Use solid lines for all curves
        
        plt.plot(days, curve, color=color, linestyle=linestyle, linewidth=2,
                label=f"{polymer} - {grade} (90d: {curve[89]:.1f}%, max: {curve[-1]:.1f}%)")
        
        print(f"Generated curve for {polymer} - {grade}: max={curve[-1]:.1f}%")
    
    plt.xlabel('Time (days)')
    plt.ylabel('Disintegration (%)')
    plt.title('Low-Disintegration Homopolymer Curves at 28°C (Zoomed: 0-5%)')
    plt.ylim(0, 5)  # Zoomed in view
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save zoomed plot
    plt.savefig('low_disintegration_homopolymers_zoomed.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved zoomed plot to: low_disintegration_homopolymers_zoomed.png")
    
    # Create figure for all curves - FULL SCALE (0-100%)
    plt.figure(figsize=(15, 10))
    
    for idx, (_, row) in enumerate(low_disintegration_sus.iterrows()):
        grade = row['Grade']
        polymer = row['Polymer Category']
        tuv_home = str(row.get('TUV Home', ''))
        thickness = parse_thickness(row.get('Thickness 1', None))
        
        # Use a unique seed for each material
        material_seed = hash(f"{polymer}_{grade}") % (2**32)
        
        # Generate curve using existing model
        curve = generate_material_curve(
            polymer, grade, tuv_home, thickness, material_seed=material_seed
        )
        
        # Plot curve with unique color
        days = range(0, DAYS)  # Start at 0 instead of 1
        is_home = is_home_compostable_certified(tuv_home)
        color = cmap(idx / len(low_disintegration_sus))  # Unique color for each curve
        linestyle = '-'  # Use solid lines for all curves
        
        plt.plot(days, curve, color=color, linestyle=linestyle, linewidth=2,
                label=f"{polymer} - {grade} (90d: {curve[89]:.1f}%, max: {curve[-1]:.1f}%)")
    
    plt.xlabel('Time (days)')
    plt.ylabel('Disintegration (%)')
    plt.title('Low-Disintegration Homopolymer Curves at 28°C (Full Scale: 0-100%)')
    plt.ylim(0, 100)  # Full scale view
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save full-scale plot
    plt.savefig('low_disintegration_homopolymers_fullscale.png', dpi=300, bbox_inches='tight')
    print(f"Saved full-scale plot to: low_disintegration_homopolymers_fullscale.png")
    
    # Also save individual curves
    for idx, (_, row) in enumerate(low_disintegration_sus.iterrows()):
        grade = row['Grade']
        polymer = row['Polymer Category']
        tuv_home = str(row.get('TUV Home', ''))
        thickness = parse_thickness(row.get('Thickness 1', None))
        
        # Use a unique seed for each material
        material_seed = hash(f"{polymer}_{grade}") % (2**32)
        
        # Generate curve
        curve = generate_material_curve(polymer, grade, tuv_home, thickness, material_seed=material_seed)
        
        # Create individual plot
        plt.figure(figsize=(10, 6))
        days = range(0, DAYS)  # Start at 0 instead of 1
        is_home = is_home_compostable_certified(tuv_home)
        color = 'green' if is_home else 'red'
        
        plt.plot(days, curve, color=color, linewidth=2)
        plt.xlabel('Time (days)')
        plt.ylabel('Disintegration (%)')
        plt.title(f'{polymer} - {grade}\nDisintegration Curve at 28°C')
        plt.ylim(0, 100)  # Set y-axis to 0-100%
        plt.grid(True, alpha=0.3)
        
        # Fix: handle NaN or float grade
        grade_str = str(grade) if pd.notna(grade) else "unknown"
        filename = f"{polymer}_{grade_str.replace(' ', '_').replace('/', '_')}_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual curve: {filename}")

if __name__ == "__main__":
    print("=== GENERATING LOW-DISINTEGRATION HOMOPOLYMER CURVES (PETROLEUM + PLA) ===")
    generate_homopolymer_curves()
    print("Done!") 