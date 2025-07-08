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

    # THIS MUST BE FIRST!
    st.set_page_config(
        page_title="Polymer Blend Disintegration Model",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Set dark background for the whole app
    st.markdown("""
    <style>
    body, .stApp, .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v0mbdj, .css-1dp5vir, .css-1cpxqw2, .css-ffhzg2, .css-1outpf7, .css-1lcbmhc, .css-1vq4p4l, .css-1wrcr25, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb, .css-1vzeuhh, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb, .css-1vzeuhh, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stMetric, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stAlert, .stSuccess, .stError, .stWarning, .stInfo, .stRadio, .stCheckbox, .stSlider, .stTextInput, .stTextArea, .stDateInput, .stTimeInput, .stColorPicker, .stFileUploader, .stImage, .stAudio, .stVideo, .stJson, .stCode, .stException, .stHelp, .stExpander, .stTabs, .stTab, .stSidebar, .stSidebarContent, .stSidebarNav, .stSidebarNavItem, .stSidebarNavLink, .stSidebarNavLinkActive, .stSidebarNavLinkInactive, .stSidebarNavLinkSelected, .stSidebarNavLinkUnselected, .stSidebarNavLinkDisabled, .stSidebarNavLinkIcon, .stSidebarNavLinkLabel, .stSidebarNavLinkLabelText, .stSidebarNavLinkLabelIcon, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Aggressive green button CSS
    st.markdown("""
    <style>
    button[kind='primary'],
    button[kind='primary']:hover,
    button[kind='primary']:focus,
    button[kind='primary']:active,
    button[kind='primary']:focus-visible,
    button[kind='primary']:focus-within,
    button[kind='primary']:focus:not(:active),
    .stButton > button[kind='primary'],
    .stButton > button[kind='primary']:hover,
    .stButton > button[kind='primary']:focus,
    .stButton > button[kind='primary']:active,
    .stButton > button[kind='primary']:focus-visible,
    .stButton > button[kind='primary']:focus-within,
    .stButton > button[kind='primary']:focus:not(:active),
    div[data-testid='stButton'] > button[kind='primary'],
    div[data-testid='stButton'] > button[kind='primary']:hover,
    div[data-testid='stButton'] > button[kind='primary']:focus,
    div[data-testid='stButton'] > button[kind='primary']:active,
    div[data-testid='stButton'] > button[kind='primary']:focus-visible,
    div[data-testid='stButton'] > button[kind='primary']:focus-within,
    div[data-testid='stButton'] > button[kind='primary']:focus:not(:active) {
        background-color: #2E8B57 !important;
        border-color: #2E8B57 !important;
        color: white !important;
        font-weight: 600 !important;
        outline: none !important;
        box-shadow: none !important;
        border: 2px solid #2E8B57 !important;
    }
    button[kind='primary']:hover,
    .stButton > button[kind='primary']:hover,
    div[data-testid='stButton'] > button[kind='primary']:hover {
        background-color: #3CB371 !important;
        border-color: #3CB371 !important;
        border: 2px solid #3CB371 !important;
    }
    button[kind='primary']:active,
    .stButton > button[kind='primary']:active,
    div[data-testid='stButton'] > button[kind='primary']:active {
        background-color: #228B22 !important;
        border-color: #228B22 !important;
        border: 2px solid #228B22 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Force all Streamlit widget labels to be white
    st.markdown("""
    <style>
    label, .stSelectbox label, .stNumberInput label, .css-1cpxqw2, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0 {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'plot_generated' not in st.session_state:
        st.session_state['plot_generated'] = False
    
    # Professional header
    st.markdown('<h1 class="main-header">🌱 Polymer Blend Disintegration Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced physics-based modeling for home-compostable polymer blends at 28°C</p>', unsafe_allow_html=True)
    
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
        home_status = " 🌿" if is_home else " ⚗️"
        display_name = f"{polymer} - {grade}{home_status}"
        
        material_options.append({
            'display': display_name,
            'polymer': polymer,
            'grade': grade,
            'tuv_home': tuv_home,
            'row_data': row
        })
    
    # Main interface - 35% left for selection, 65% right for plot
    col1, col2 = st.columns([35, 65])
    
    with col1:
        st.markdown("### Material Selection")
        
        selected_materials = []
        volume_fractions = []
        
        # Material selection interface
        for i in range(5):
            cols = st.columns([3, 1])
            
            with cols[0]:
                material_selection = st.selectbox(
                    f"Material {i+1}",
                    options=[""] + [opt['display'] for opt in material_options],
                    key=f"material_{i}"
                )
            
            with cols[1]:
                vol_frac = st.number_input(
                    "Fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=f"vol_frac_{i}"
                )
            
            if material_selection and vol_frac > 0:
                selected_material = next(opt for opt in material_options if opt['display'] == material_selection)
                selected_materials.append(selected_material)
                volume_fractions.append(vol_frac)
        
        # Volume fraction validation
        total_vol_frac = sum(volume_fractions)
        if selected_materials:
            if abs(total_vol_frac - 1.0) <= 0.01:
                st.success(f"✅ Total: {total_vol_frac:.2f}")
            else:
                st.error(f"❌ Total: {total_vol_frac:.2f} (should be 1.0)")
        
        # Generate button
        if st.button("🚀 Generate Disintegration Curve", type="primary"):
            if not selected_materials:
                st.error("Please select at least one material.")
            elif abs(total_vol_frac - 1.0) > 0.01:
                st.error("Volume fractions must sum to 1.0.")
            else:
                # Create blend string
                blend_parts = []
                for material, vol_frac in zip(selected_materials, volume_fractions):
                    blend_parts.extend([material['grade'], str(vol_frac)])
                blend_string = ",".join(blend_parts)
                
                try:
                    # Generate curve
                    temp_output = "temp_blend_curve.png"
                    generate_custom_blend_curves([blend_string], temp_output)
                    
                    # Store the result for display in the right column
                    st.session_state['plot_generated'] = True
                    st.session_state['temp_output'] = temp_output
                    st.session_state['selected_materials'] = selected_materials
                    st.session_state['volume_fractions'] = volume_fractions
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        

    
    with col2:
        st.markdown("### Results")
        
        # Check if plot has been generated
        if 'plot_generated' in st.session_state and st.session_state['plot_generated']:
            # Display plot
            st.image(st.session_state['temp_output'], use_container_width=True)
            
            # Download button
            with open(st.session_state['temp_output'], "rb") as file:
                st.download_button(
                    label="📥 Download Plot",
                    data=file.read(),
                    file_name="polymer_blend_disintegration.png",
                    mime="image/png"
                )
            
            # Show only 90 day and 180 day values (for the first/only blend)
            # Get the blend curve from the file or from the last calculation
            # We'll recalculate it here for robustness
            blend_parts = []
            for material, vol_frac in zip(st.session_state['selected_materials'], st.session_state['volume_fractions']):
                blend_parts.extend([material['grade'], str(vol_frac)])
            blend_string = ",".join(blend_parts)
            from modules.blend_generator import generate_blend
            _, blend_curve = generate_blend(blend_string)
            
            value_90 = blend_curve[89] if len(blend_curve) > 89 else float('nan')
            value_180 = blend_curve[179] if len(blend_curve) > 179 else float('nan')
            
            st.markdown(f"""
            <div style='margin-top:2em;font-size:1.3em;'>
                <b>mid-point (90 day):</b> {value_90:.1f}%<br>
                <b>max (180 day):</b> {value_180:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            **Select materials and generate a curve to see results here.**
            
            The plot will appear in this area once you click "Generate Disintegration Curve".
            """)

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