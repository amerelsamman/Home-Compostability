"""
Generate time-series curves for 90:10 home-nonhome blends for QA purposes.
Uses the synergistic boost model from modules.
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
    generate_material_curve, generate_material_curve_with_synergistic_boost, 
    parse_thickness, find_material_by_grade, is_home_compostable_certified
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
    
    return home_certified

def filter_medium_disintegration_materials(sus_data):
    """Filter sustainability data to include only medium-disintegration polymers (non-home, non-petroleum, non-PLA)"""
    print(f"Total materials in dataset: {len(sus_data)}")
    
    # First, exclude home-compostable certified materials
    # Only exclude rows where 'TUV Home' is exactly 'Certified' (case-insensitive)
    home_certified = sus_data[sus_data['TUV Home'].str.strip().str.lower() == 'certified']
    
    non_home = sus_data[~sus_data.index.isin(home_certified.index)]
    print(f"Materials after excluding home-compostable: {len(non_home)}")
    
    # Then exclude low disintegration polymers (petroleum-based + PLA) using core_model logic
    # This matches the is_low_disintegration_polymer function in core_model.py
    low_disintegration_keywords = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'BIO-PE', 'PLA']
    
    def is_low_disintegration(polymer):
        if pd.isna(polymer):
            return False
        return any(keyword in str(polymer).upper() for keyword in low_disintegration_keywords)
    
    # Use 'Polymer Category' column to match the actual data structure
    medium_disintegration = non_home[~non_home['Polymer Category'].apply(is_low_disintegration)]
    print(f"Materials after excluding low disintegration: {len(medium_disintegration)}")
    
    # Show what polymer types we found
    polymer_types = medium_disintegration['Polymer Category'].value_counts()
    print(f"Polymer types found: {polymer_types.to_dict()}")
    
    return medium_disintegration

def generate_blend_curves():
    """Generate time-series curves for all 90:10 home-nonhome blends"""
    
    # Filter for home-compostable and medium-disintegration materials
    home_compostable_sus = filter_home_compostable_materials(sus)
    medium_disintegration_sus = filter_medium_disintegration_materials(sus)
    
    print(f"Found {len(home_compostable_sus)} home-compostable certified materials")
    print(f"Found {len(medium_disintegration_sus)} medium-disintegration materials")
    print(f"Total possible 90:10 blends: {len(home_compostable_sus) * len(medium_disintegration_sus)}")
    
    # Create figure for all blend curves
    plt.figure(figsize=(15, 10))
    
    # Get a color map for unique colors
    cmap = plt.cm.get_cmap('tab20')
    
    blend_count = 0
    for home_idx, (_, home_row) in enumerate(home_compostable_sus.iterrows()):
        home_grade = home_row['Grade']
        home_polymer = home_row['Polymer Category']
        home_tuv_home = str(home_row.get('TUV Home', ''))
        home_thickness = parse_thickness(home_row.get('Thickness 1', None))
        
        for nonhome_idx, (_, nonhome_row) in enumerate(medium_disintegration_sus.iterrows()):
            nonhome_grade = nonhome_row['Grade']
            nonhome_polymer = nonhome_row['Polymer Category']
            nonhome_tuv_home = str(nonhome_row.get('TUV Home', ''))
            nonhome_thickness = parse_thickness(nonhome_row.get('Thickness 1', None))
            
            # Use a unique seed for each blend
            material_seed = hash(f"{home_polymer}_{home_grade}_{nonhome_polymer}_{nonhome_grade}") % (2**32)
            
            # Generate synergistic curve for the non-home component (90% home, 10% non-home)
            home_fraction = 0.9  # 90% home-compostable
            curve = generate_material_curve_with_synergistic_boost(
                nonhome_polymer, nonhome_grade, nonhome_tuv_home, nonhome_thickness, 
                home_fraction, material_seed=material_seed
            )
            
            # Plot curve with unique color
            days = range(0, DAYS)
            color = cmap(blend_count / (len(home_compostable_sus) * len(medium_disintegration_sus)))
            
            plt.plot(days, curve, color=color, linewidth=2,
                    label=f"90%{home_polymer}-10%{nonhome_polymer} (90d: {curve[89]:.1f}%, max: {curve[-1]:.1f}%)")
            
            print(f"Generated blend {blend_count+1}: 90%{home_polymer}-10%{nonhome_polymer}: max={curve[-1]:.1f}%")
            blend_count += 1
    
    plt.xlabel('Time (days)')
    plt.ylabel('Disintegration (%)')
    plt.title('90:10 Home-Nonhome Blend Curves at 28°C (Synergistic Boost Effect)')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('home_nonhome_blends.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: home_nonhome_blends.png")

if __name__ == "__main__":
    print("=== GENERATING 90:10 HOME-NONHOME BLEND CURVES (SYNERGISTIC BOOST EFFECT) ===")
    generate_blend_curves()
    print("Done!") 