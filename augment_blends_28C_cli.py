"""
Command-line interface for polymer blend disintegration model.
Orchestrates the modules to provide CLI functionality for generating blend curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.stats import dirichlet

# Import functions from modules
from modules.core_model import (
    sus, DAYS, SAMPLE_AREA, THICKNESS_DEFAULT,
    generate_material_curve, parse_thickness, find_material_by_grade,
    parse_blend_input, is_home_compostable_certified, sigmoid, get_max_disintegration_hybrid
)
from modules.blend_generator import generate_blend, generate_csv_for_single_blend, generate_random_blends
from modules.plotting import generate_custom_blend_curves

# --- COMMAND LINE ARGUMENT PARSING ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate polymer blend disintegration curves at 28°C')
    parser.add_argument('--blends', nargs='+', help='Custom blends in format: "grade1,vol_frac1,grade2,vol_frac2,..." (up to 5 materials)')
    parser.add_argument('--output', default='custom_blend_curve.png', help='Output PNG filename (default: custom_blend_curve.png)')
    parser.add_argument('--predefined', action='store_true', help='Run with predefined blend examples instead of custom blends')
    parser.add_argument('--random', action='store_true', help='Generate random blends using Dirichlet distribution')
    parser.add_argument('--num_blends', type=int, default=100, help='Number of random blends to generate (default: 100)')
    parser.add_argument('--max_materials', type=int, default=3, help='Maximum number of materials per blend (default: 3)')
    parser.add_argument('--csv_output', default='random_blends_28C.csv', help='CSV output filename for random blends (default: random_blends_28C.csv)')
    return parser.parse_args()

# --- STREAMLIT APP ---
def run_streamlit_app():
    import streamlit as st
    
    st.set_page_config(page_title="Polymer Blend Disintegration Generator", layout="wide")
    
    st.title("🎯 Polymer Blend Disintegration Generator at 28°C")
    st.markdown("Generate disintegration curves for custom polymer blends using the same physics as the CLI version.")
    
    # Load available materials
    material_options = []
    for _, row in sus.iterrows():
        grade_value = row['Grade']
        if pd.isna(grade_value):
            continue
        
        polymer = str(row.get('Polymer Category', ''))
        grade = str(grade_value)
        tuv_home = str(row.get('TUV Home', ''))
        
        # Check if home-compostable
        is_home = is_home_compostable_certified(tuv_home)
        home_status = " (HOME)" if is_home else " (NON-HOME)"
        display_name = f"{polymer} - {grade}{home_status}"
        
        material_options.append({
            'display': display_name,
            'polymer': polymer,
            'grade': grade,
            'tuv_home': tuv_home,
            'row_data': row
        })
    
    # Create material selection interface
    st.header("📋 Material Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select Materials (up to 5)")
        
        selected_materials = []
        volume_fractions = []
        
        for i in range(5):
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                material_key = f"material_{i}"
                material_selection = st.selectbox(
                    f"Material {i+1}",
                    options=[""] + [opt['display'] for opt in material_options],
                    key=material_key
                )
            
            with col_b:
                vol_frac_key = f"vol_frac_{i}"
                vol_frac = st.number_input(
                    "Volume Fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=vol_frac_key
                )
            
            if material_selection and vol_frac > 0:
                selected_material = next(opt for opt in material_options if opt['display'] == material_selection)
                selected_materials.append(selected_material)
                volume_fractions.append(vol_frac)
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. **Select up to 5 materials** from the dropdowns
        2. **Set volume fractions** (should sum to 1.0)
        3. **Click Generate** to see the blend curve
        
        """)
    
    # Validate volume fractions
    total_vol_frac = sum(volume_fractions)
    if selected_materials and abs(total_vol_frac - 1.0) > 0.01:
        st.warning(f"⚠️ Volume fractions should sum to 1.0 (currently: {total_vol_frac:.2f})")
    
    # Generate button
    if st.button("🚀 Generate Blend Curve", type="primary"):
        if not selected_materials:
            st.error("Please select at least one material.")
        elif abs(total_vol_frac - 1.0) > 0.01:
            st.error("Volume fractions must sum to 1.0.")
        else:
            # Create blend string for the CLI function
            blend_parts = []
            for material, vol_frac in zip(selected_materials, volume_fractions):
                blend_parts.extend([material['grade'], str(vol_frac)])
            
            blend_string = ",".join(blend_parts)
            
            # Generate the curve using the same function as CLI
            try:
                # Create a temporary output file
                temp_output = "temp_blend_curve.png"
                generate_custom_blend_curves([blend_string], temp_output)
                
                # Display results
                st.success("✅ Blend curve generated successfully!")
                
                # Show the plot
                st.image(temp_output, caption="Generated Blend Curve", use_container_width=True)
                
                # Display material information
                st.subheader("📊 Material Information")
                for i, (material, vol_frac) in enumerate(zip(selected_materials, volume_fractions)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**{material['polymer']} {material['grade']}**")
                        st.write(f"Volume Fraction: {vol_frac:.1%}")
                    
                    with col2:
                        is_home = is_home_compostable_certified(material['tuv_home'])
                        st.write(f"Home-compostable: {'Yes' if is_home else 'No'}")
                    
                    with col3:
                        thickness_val = parse_thickness(material['row_data'].get('Thickness 1', None))
                        st.write(f"Thickness: {thickness_val:.3f} mm")
                    
                    st.divider()
                
            except Exception as e:
                st.error(f"Error generating blend curve: {str(e)}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.random:
        # Generate random blends and save to CSV
        print("=== RANDOM BLEND GENERATION MODE ===")
        blend_data = generate_random_blends(args.num_blends, args.max_materials)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(blend_data)
        df.to_csv(args.csv_output, index=False)
        print(f"\n✅ Generated {args.num_blends} random blends with {len(blend_data)} total data points")
        print(f"📁 Saved to: {args.csv_output}")
        
        # Show summary statistics
        home_blends = df[df['home_status'] == 'home']['blend_label'].nunique()
        non_home_blends = df[df['home_status'] == 'non-home']['blend_label'].nunique()
        print(f"📊 Summary:")
        print(f"   - Home-compostable blends: {home_blends}")
        print(f"   - Non-home-compostable blends: {non_home_blends}")
        print(f"   - Total unique blends: {df['blend_label'].nunique()}")
        
    elif args.blends or args.predefined:
        # Original CLI functionality
        if args.predefined:
            # Run with predefined examples
            predefined_blends = [
                "4032D,0.7,ecoflex® F Blend C1200,0.3",
                "PHACT S1000P,0.5,PHACT A1000P,0.5",
                "Bioplast GF 106,0.6,Bioplast 500 A,0.4"
            ]
            generate_custom_blend_curves(predefined_blends, args.output)
        else:
            # Run with custom blends
            generate_custom_blend_curves(args.blends, args.output)
    else:
        # Default to Streamlit app
        run_streamlit_app() 