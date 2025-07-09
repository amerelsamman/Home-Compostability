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

def filter_home_compostable_materials(sus_data):
    """Filter sustainability data to include only home-compostable certified polymers"""
    print(f"Total materials in dataset: {len(sus_data)}")
    
    # Include ONLY home-compostable certified materials
    # Only include rows where 'TUV Home' is exactly 'Certified' (case-insensitive)
    home_certified = sus_data[sus_data['TUV Home'].str.strip().str.lower() == 'certified']
    print(f"Home-compostable certified materials found: {len(home_certified)}")
    
    # Show what polymer types we found
    polymer_types = home_certified['Polymer Category'].value_counts()
    print(f"Polymer types found: {polymer_types.to_dict()}")
    
    # Show some examples of what we're including
    print("\nExamples of home-compostable materials:")
    for _, row in home_certified.head(10).iterrows():
        print(f"  - {row['Polymer Category']} ({row['Grade']}) - Max disintegration: {row.get('Max Disintegration', 'N/A')}")
    
    return home_certified

def generate_homopolymer_curves():
    """Generate time-series curves for all home-compostable certified homopolymers"""
    
    # Filter for home-compostable materials
    home_compostable_sus = filter_home_compostable_materials(sus)
    print(f"Found {len(home_compostable_sus)} home-compostable certified materials")
    
    # Create figure for all curves - FULL SCALE (0-100%)
    plt.figure(figsize=(15, 10))
    
    # Get a color map for unique colors
    cmap = plt.cm.get_cmap('tab20')
    
    for idx, (_, row) in enumerate(home_compostable_sus.iterrows()):
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
        color = cmap(idx / len(home_compostable_sus))  # Unique color for each curve
        linestyle = '-'  # Use solid lines for all curves
        
        plt.plot(days, curve, color=color, linestyle=linestyle, linewidth=2,
                label=f"{polymer} - {grade} (90d: {curve[89]:.1f}%, max: {curve[-1]:.1f}%)")
    
    plt.xlabel('Time (days)')
    plt.ylabel('Disintegration (%)')
    plt.title('Home-Compostable Certified Homopolymer Curves at 28°C')
    plt.ylim(0, 100)  # Full scale view
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('home_compostable_homopolymers.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: home_compostable_homopolymers.png")
    
    # Also save individual curves
    for idx, (_, row) in enumerate(home_compostable_sus.iterrows()):
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
        
        # Fix: handle NaN or float grade and special characters in filename
        grade_str = str(grade) if pd.notna(grade) else "unknown"
        # Replace problematic characters for filenames
        safe_polymer = polymer.replace('/', '_').replace(' ', '_').replace('®', '').replace('™', '')
        safe_grade = grade_str.replace(' ', '_').replace('/', '_').replace('®', '').replace('™', '').replace('°', '')
        filename = f"{safe_polymer}_{safe_grade}_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual curve: {filename}")

if __name__ == "__main__":
    print("=== GENERATING HOME-COMPOSTABLE CERTIFIED HOMOPOLYMER CURVES ===")
    generate_homopolymer_curves()
    print("Done!") 