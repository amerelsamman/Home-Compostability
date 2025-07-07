import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .core_model import (
    sus, DAYS, SAMPLE_AREA, THICKNESS_DEFAULT,
    generate_material_curve, parse_thickness, find_material_by_grade,
    parse_blend_input, is_home_compostable_certified, sigmoid, get_max_disintegration_hybrid
)
from .blend_generator import generate_blend

def generate_custom_blend_curves(blend_strings, output_filename):
    """Generate curves for custom blends"""
    print("Generating custom blend curves...")
    
    custom_blend_curves = []
    custom_blend_labels = []
    
    for i, blend_str in enumerate(blend_strings):
        try:
            print(f"\nProcessing custom blend {i+1}: {blend_str}")
            
            # Use the core blend generation function
            material_info, blend_curve = generate_blend(blend_str)
            
            custom_blend_curves.append(blend_curve)
            
            # Create label
            label_parts = []
            for material in material_info:
                label_parts.append(f"{material['polymer']} {material['grade']} ({material['vol_frac']:.1%})")
            custom_blend_labels.append(" + ".join(label_parts))
            
            print(f"  Generated blend curve with max disintegration: {np.max(blend_curve):.1f}%")
                
        except ValueError as e:
            print(f"ERROR: {e}")
            return
    
    # Plot custom blends
    if custom_blend_curves:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (curve, label) in enumerate(zip(custom_blend_curves, custom_blend_labels)):
            color = colors[i % len(colors)]
            plt.plot(np.arange(1, DAYS+1), curve, label=label, linewidth=3, color=color)
            
            # Add max disintegration label
            max_disintegration = np.max(curve)
            max_day = np.argmax(curve) + 1  # +1 because days start at 1
            plt.annotate(f'Max: {max_disintegration:.1f}%', 
                        xy=(max_day, max_disintegration), 
                        xytext=(max_day + 10, max_disintegration + 2),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.7),
                        fontsize=10, color=color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add 90-day mark label
            if 90 <= len(curve):
                day_90_value = curve[89]  # Index 89 corresponds to day 90
                plt.annotate(f'90d: {day_90_value:.1f}%', 
                            xy=(90, day_90_value), 
                            xytext=(90 + 5, day_90_value - 3),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=0.7),
                            fontsize=9, color=color, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9))
            
            print(f"Plotted: {label} (max: {max_disintegration:.1f}%, 90d: {day_90_value:.1f}%)")
        
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Disintegration (%)', fontsize=12)
        plt.title('Custom Polymer Blend Disintegration Curves at 28°C', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)  # Set y-axis from 0 to 105 to show full asymptotic behavior
        plt.tight_layout()
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nCustom blend plot saved as {output_filename}")
        plt.show()
    else:
        print("No custom blend curves to plot!") 