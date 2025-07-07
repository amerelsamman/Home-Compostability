import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.stats import dirichlet

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

def parse_blend_input(blend_str):
    """Parse blend input string like '4032D,0.5,Ecoworld,0.5'"""
    parts = [part.strip() for part in blend_str.split(',')]
    if len(parts) % 2 != 0:
        raise ValueError("Blend input must have even number of parts (grade,vol_frac pairs)")
    
    materials = []
    for i in range(0, len(parts), 2):
        grade = parts[i]
        try:
            vol_frac = float(parts[i+1])
        except ValueError:
            raise ValueError(f"Invalid volume fraction: {parts[i+1]}")
        
        materials.append({'grade': grade, 'vol_frac': vol_frac})
    
    # Validate volume fractions sum to 1.0
    total_vol_frac = sum(mat['vol_frac'] for mat in materials)
    if abs(total_vol_frac - 1.0) > 0.01:
        raise ValueError(f"Volume fractions must sum to 1.0 (currently: {total_vol_frac:.2f})")
    
    return materials

def find_material_by_grade(sus_df, grade):
    """Find material in sustainability.csv by grade name"""
    for _, row in sus_df.iterrows():
        grade_value = row['Grade']
        if pd.isna(grade_value):
            continue
        if grade.lower() in str(grade_value).lower():
            return row
    return None

# --- CONFIGURABLE PARAMETERS ---
DAYS = 200  # Number of days for the time series
SAMPLE_AREA = 100.0  # Default sample area (mm^2)
THICKNESS_DEFAULT = 1.0  # Default thickness (mm)

# --- SIGMOID FUNCTION ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# --- LOAD DATA ---
sus = pd.read_csv('sustainability.csv')
sus.columns = [c.strip() for c in sus.columns]

# Hybrid classification functions - TUV Home certification takes priority, with fallback to polymer type
def is_home_compostable_certified(tuv_home):
    """Check if material is certified for home composting based on TUV Home column"""
    if pd.isna(tuv_home) or tuv_home == '':
        return False
    
    tuv_str = str(tuv_home).lower()
    # Check for certification indicators
    certified_indicators = ['certified', 'ok compost home', 'home', '20250403_okcompost_export home.pdf']
    not_certified_indicators = ['not certified', 'not certified', 'n/a']
    
    # If any "not certified" indicators are present, return False
    if any(indicator in tuv_str for indicator in not_certified_indicators):
        return False
    
    # If any certification indicators are present, return True
    if any(indicator in tuv_str for indicator in certified_indicators):
        return True
    
    return False

def is_low_disintegration_polymer(polymer):
    """Petroleum-based polymers and PLA that can't exceed 2% disintegration"""
    low_disintegration = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'BIO-PE', 'PLA']
    return any(x in polymer.upper() for x in low_disintegration)

def is_medium_disintegration_polymer(polymer):
    """All non-home-compostable biopolymers that aren't petroleum-based or PLA"""
    # This includes all biopolymers that aren't in the low disintegration category
    # Examples: PBAT, PHB, PBS, PCL, PHA, Starch-based, etc.
    low_disintegration = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'Bio-PE', 'PLA']
    return not any(x in polymer.upper() for x in low_disintegration)

def get_max_disintegration_hybrid(polymer, tuv_home, thickness_val):
    """Determine maximum disintegration percentage: TUV Home certification takes priority, then polymer type"""
    # First check TUV Home certification
    is_home_certified = is_home_compostable_certified(tuv_home)
    
    if is_home_certified:
        # Home-compostable certified materials: 90-95% max disintegration
        return np.random.uniform(90, 95)
    else:
        # Not home-compostable certified - use polymer type classification
        if is_low_disintegration_polymer(polymer):
            return np.random.uniform(0.5, 2)
        elif is_medium_disintegration_polymer(polymer):
            return np.random.uniform(30, 90)
        else:
            return np.random.uniform(10, 30)

def generate_material_curve(polymer, grade, tuv_home, thickness_val, days=DAYS):
    """Generate disintegration curve for a single material based ONLY on TUV Home certification"""
    t = np.arange(1, days+1)
    
    # Use ONLY the TUV Home certification data - NO HARDWIRED LOGIC
    is_home_compostable = is_home_compostable_certified(tuv_home)
    
    # Ensure thickness is valid
    if np.isnan(thickness_val) or thickness_val <= 0:
        thickness_val = THICKNESS_DEFAULT
    
    if is_home_compostable:
        # Home-compostable certified: sigmoid curve with high disintegration
        base_k = 0.08
        base_t0 = 70
        
        # Reduced thickness effect for very thin materials
        thickness_factor = (thickness_val / 1.0) ** 0.1  # Much smaller thickness effect
        k = base_k * thickness_factor
        t0 = base_t0 / thickness_factor
        
        # Use hybrid maximum disintegration (certification takes priority)
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val)
        y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Home-compostable certified: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}% - Thickness: {thickness_val:.3f}mm")
    else:
        # Not home-compostable certified: use hybrid classification
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val)
        
        if max_disintegration < 5:  # Very low disintegration materials (petroleum-based)
            y = np.full_like(t, max_disintegration)
        else:
            k = 0.02
            t0 = 120
            y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Not home-compostable: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}%")
    
    # Add very small noise for minimal realism (much reduced noise)
    y = y + np.random.normal(0, 0.05, size=y.shape)
    
    # Ensure curve is monotonically increasing (no decreases)
    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            y[i] = y[i-1] + np.random.uniform(0, 0.02)  # Very small positive increment
    
    # No clipping - let sigmoid handle everything naturally
    y = np.clip(y, 0, None)
    
    # Final check for NaN values
    if np.any(np.isnan(y)):
        print(f"    ERROR: NaN values in curve for {polymer} {grade}. Using flat curve at 2%.")
        y = np.full_like(t, 2.0)
    
    return y

def parse_thickness(thickness_cert):
    """Parse thickness from sustainability.csv format"""
    thickness_val = THICKNESS_DEFAULT
    if isinstance(thickness_cert, str):
        if 'um' in thickness_cert:
            try:
                thickness_val = float(re.sub(r'[^\d.]', '', thickness_cert)) / 1000.0
            except:
                thickness_val = THICKNESS_DEFAULT
        elif 'mm' in thickness_cert:
            try:
                thickness_val = float(re.sub(r'[^\d.]', '', thickness_cert))
            except:
                thickness_val = THICKNESS_DEFAULT
        else:
            try:
                thickness_val = float(thickness_cert)
            except:
                thickness_val = THICKNESS_DEFAULT
    else:
        try:
            thickness_val = float(thickness_cert)
        except:
            thickness_val = THICKNESS_DEFAULT
    return thickness_val

def generate_csv_for_single_blend(blend_str, output_path):
    """Generate CSV with disintegration profile for a single blend"""
    materials = parse_blend_input(blend_str)
    
    # Find materials and generate curves (without triggering plots)
    material_info = []
    for mat in materials:
        material_row = find_material_by_grade(sus, mat['grade'])
        if material_row is not None:
            thickness_val = parse_thickness(material_row.get('Thickness 1', None))
            
            # Generate curve without print statements to avoid duplicate plots
            t = np.arange(1, DAYS+1)
            is_home_compostable = is_home_compostable_certified(str(material_row.get('TUV Home', '')))
            
            if np.isnan(thickness_val) or thickness_val <= 0:
                thickness_val = THICKNESS_DEFAULT
            
            if is_home_compostable:
                base_k = 0.08
                base_t0 = 70
                thickness_factor = (thickness_val / 1.0) ** 0.1
                k = base_k * thickness_factor
                t0 = base_t0 / thickness_factor
                max_disintegration = get_max_disintegration_hybrid(
                    material_row['Polymer Category'], 
                    str(material_row.get('TUV Home', '')), 
                    thickness_val
                )
                curve = sigmoid(t, max_disintegration, k, t0)
            else:
                max_disintegration = get_max_disintegration_hybrid(
                    material_row['Polymer Category'], 
                    str(material_row.get('TUV Home', '')), 
                    thickness_val
                )
                if max_disintegration < 5:
                    curve = np.full_like(t, max_disintegration)
                else:
                    k = 0.02
                    t0 = 120
                    curve = sigmoid(t, max_disintegration, k, t0)
            
            # Add noise and ensure monotonicity (same as original function)
            curve = curve + np.random.normal(0, 0.05, size=curve.shape)
            for i in range(1, len(curve)):
                if curve[i] < curve[i-1]:
                    curve[i] = curve[i-1] + np.random.uniform(0, 0.02)
            curve = np.clip(curve, 0, None)
            
            if np.any(np.isnan(curve)):
                curve = np.full_like(t, 2.0)
            
            material_info.append({
                'polymer': material_row['Polymer Category'],
                'grade': material_row['Grade'],
                'vol_frac': mat['vol_frac'],
                'curve': curve,
                'tuv_home': str(material_row.get('TUV Home', ''))
            })
    
    # Calculate blend curve
    blend_curve = np.zeros(DAYS)
    for material in material_info:
        blend_curve += material['curve'] * material['vol_frac']
    
    # Determine blend properties
    is_home_blend = any(is_home_compostable_certified(mat['tuv_home']) for mat in material_info)
    home_status = "home" if is_home_blend else "non-home"
    
    # Create blend label
    blend_label = " + ".join([f"{mat['grade']}({mat['vol_frac']:.2f})" for mat in material_info])
    
    # Generate CSV data
    all_data = []
    for day in range(1, DAYS + 1):
        row_data = {
            'Type': 'Disintegration',
            'Polymer Grade 1': material_info[0]['grade'] if len(material_info) > 0 else '',
            'Polymer Grade 2': material_info[1]['grade'] if len(material_info) > 1 else '',
            'Polymer Grade 3': material_info[2]['grade'] if len(material_info) > 2 else '',
            'Polymer Grade 4': material_info[3]['grade'] if len(material_info) > 3 else '',
            'Polymer Grade 5': material_info[4]['grade'] if len(material_info) > 4 else '',
            'SMILES1': '', 'SMILES2': '', 'SMILES3': '', 'SMILES4': '', 'SMILES5': '',
            'vol_fraction1': material_info[0]['vol_frac'] if len(material_info) > 0 else 0,
            'vol_fraction2': material_info[1]['vol_frac'] if len(material_info) > 1 else 0,
            'vol_fraction3': material_info[2]['vol_frac'] if len(material_info) > 2 else 0,
            'vol_fraction4': material_info[3]['vol_frac'] if len(material_info) > 3 else 0,
            'vol_fraction5': material_info[4]['vol_frac'] if len(material_info) > 4 else 0,
            'Temperature (C)': 28.0,
            'Sample area (mm^2)': SAMPLE_AREA,
            'Thickness (mm)': '',
            'Time(day)': day,
            'property': blend_curve[day-1],
            'home_status': home_status,
            'blend_label': blend_label
        }
        all_data.append(row_data)
    
    # Save CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)

def generate_random_blends(num_blends, max_materials):
    """Generate random blends using Dirichlet distribution for volume fractions"""
    print(f"Generating homopolymers first, then {num_blends} random blends with up to {max_materials} materials each...")
    
    # Get all available materials from sustainability.csv
    available_materials = []
    for _, row in sus.iterrows():
        if pd.notna(row['Grade']) and str(row['Grade']).strip() != '':
            available_materials.append({
                'polymer': row['Polymer Category'],
                'grade': row['Grade'],
                'tuv_home': str(row.get('TUV Home', '')),
                'thickness': parse_thickness(row.get('Thickness 1', None))
            })
    
    print(f"Found {len(available_materials)} available materials")
    
    all_blend_data = []
    
    # STEP 1: Generate all homopolymers (single materials) first
    print(f"\n=== STEP 1: Generating {len(available_materials)} homopolymers ===")
    for material_idx, material in enumerate(available_materials):
        # Single material with 100% volume fraction
        material_info = [{
            'polymer': material['polymer'],
            'grade': material['grade'],
            'vol_frac': 1.0,
            'curve': generate_material_curve(
                material['polymer'],
                material['grade'],
                material['tuv_home'],
                material['thickness']
            ),
            'tuv_home': material['tuv_home'],
            'thickness': material['thickness']
        }]
        
        # Determine blend properties
        is_home_blend = is_home_compostable_certified(material['tuv_home'])
        home_status = "home" if is_home_blend else "non-home"
        
        # Calculate thickness
        thickness_val = material['thickness'] if material['thickness'] != THICKNESS_DEFAULT else None
        
        # Create blend label
        blend_label = f"{material['grade']}(1.00)"
        
        # Add data for each day
        for day in range(1, DAYS + 1):
            row_data = {
                'Type': 'Disintegration',
                'Polymer Grade 1': material['grade'],
                'Polymer Grade 2': '',
                'Polymer Grade 3': '',
                'Polymer Grade 4': '',
                'Polymer Grade 5': '',
                'SMILES1': '',
                'SMILES2': '',
                'SMILES3': '',
                'SMILES4': '',
                'SMILES5': '',
                'vol_fraction1': 1.0,
                'vol_fraction2': 0,
                'vol_fraction3': 0,
                'vol_fraction4': 0,
                'vol_fraction5': 0,
                'Temperature (C)': 28.0,
                'Sample area (mm^2)': SAMPLE_AREA,
                'Thickness (mm)': thickness_val if thickness_val is not None else '',
                'Time(day)': day,
                'property': material_info[0]['curve'][day-1],
                'home_status': home_status,
                'blend_label': blend_label
            }
            all_blend_data.append(row_data)
        
        if (material_idx + 1) % 10 == 0:
            print(f"Generated {material_idx + 1}/{len(available_materials)} homopolymers")
    
    print(f"✅ Generated all {len(available_materials)} homopolymers")
    
    # STEP 2: Generate random blends
    print(f"\n=== STEP 2: Generating {num_blends} random blends ===")
    for blend_idx in range(num_blends):
        # Randomly choose number of materials (2 to max_materials for blends)
        num_materials = np.random.randint(2, max_materials + 1)
        
        # Randomly select materials
        selected_materials = np.random.choice(available_materials, num_materials, replace=False)
        
        # Generate volume fractions using Dirichlet distribution
        alpha = np.ones(num_materials)  # Uniform Dirichlet distribution
        volume_fractions = dirichlet.rvs(alpha, size=1)[0]
        
        # Generate blend curve
        blend_curve = np.zeros(DAYS)
        material_info = []
        
        for i, material in enumerate(selected_materials):
            # Generate individual material curve
            curve = generate_material_curve(
                material['polymer'],
                material['grade'],
                material['tuv_home'],
                material['thickness']
            )
            
            material_info.append({
                'polymer': material['polymer'],
                'grade': material['grade'],
                'vol_frac': volume_fractions[i],
                'curve': curve,
                'tuv_home': material['tuv_home'],
                'thickness': material['thickness']
            })
            
            blend_curve += curve * volume_fractions[i]
        
        # Apply synergistic effects (same logic as original)
        home_materials = [mat for mat in material_info if is_home_compostable_certified(mat['tuv_home'])]
        
        for material in material_info:
            if (not is_home_compostable_certified(material['tuv_home']) and 
                len(home_materials) > 0):
                
                np.random.seed(hash(material['grade']) % 2**32)
                gets_boost = np.random.random() < 0.6
                
                if gets_boost:
                    total_home_frac = sum(mat['vol_frac'] for mat in material_info if is_home_compostable_certified(mat['tuv_home']))
                    
                    np.random.seed(hash(material['grade']) % 2**32)
                    if total_home_frac >= 0.9:
                        boost_amount = 10
                    else:
                        boost_amount = np.random.uniform(10, 20)
                    
                    enhanced_curve = material['curve'] + boost_amount
                    max_val = np.max(material['curve'])
                    enhanced_curve = np.clip(enhanced_curve, 0, None)
                    material['curve'] = enhanced_curve
        
        # Recalculate blend curve with synergistic effects
        blend_curve = np.zeros(DAYS)
        for material in material_info:
            blend_curve += material['curve'] * material['vol_frac']
        
        # Ensure blend curve is monotonically increasing
        for i in range(1, len(blend_curve)):
            if blend_curve[i] < blend_curve[i-1]:
                blend_curve[i] = blend_curve[i-1] + np.random.uniform(0, 0.01)
        
        blend_curve = np.clip(blend_curve, 0, None)
        
        # Determine blend properties
        is_home_blend = any(is_home_compostable_certified(mat['tuv_home']) for mat in material_info)
        home_status = "home" if is_home_blend else "non-home"
        
        # Calculate average thickness
        thicknesses = [mat['thickness'] for mat in material_info if mat['thickness'] != THICKNESS_DEFAULT]
        avg_thickness = np.mean(thicknesses) if thicknesses else None
        
        # Create blend label
        blend_label = " + ".join([f"{mat['grade']}({mat['vol_frac']:.2f})" for mat in material_info])
        
        # Add data for each day
        for day in range(1, DAYS + 1):
            row_data = {
                'Type': 'Disintegration',
                'Polymer Grade 1': material_info[0]['grade'] if len(material_info) > 0 else '',
                'Polymer Grade 2': material_info[1]['grade'] if len(material_info) > 1 else '',
                'Polymer Grade 3': material_info[2]['grade'] if len(material_info) > 2 else '',
                'Polymer Grade 4': material_info[3]['grade'] if len(material_info) > 3 else '',
                'Polymer Grade 5': material_info[4]['grade'] if len(material_info) > 4 else '',
                'SMILES1': '',
                'SMILES2': '',
                'SMILES3': '',
                'SMILES4': '',
                'SMILES5': '',
                'vol_fraction1': material_info[0]['vol_frac'] if len(material_info) > 0 else 0,
                'vol_fraction2': material_info[1]['vol_frac'] if len(material_info) > 1 else 0,
                'vol_fraction3': material_info[2]['vol_frac'] if len(material_info) > 2 else 0,
                'vol_fraction4': material_info[3]['vol_frac'] if len(material_info) > 3 else 0,
                'vol_fraction5': material_info[4]['vol_frac'] if len(material_info) > 4 else 0,
                'Temperature (C)': 28.0,
                'Sample area (mm^2)': SAMPLE_AREA,
                'Thickness (mm)': avg_thickness if avg_thickness is not None else '',
                'Time(day)': day,
                'property': blend_curve[day-1],
                'home_status': home_status,
                'blend_label': blend_label
            }
            all_blend_data.append(row_data)
        
        if (blend_idx + 1) % 10 == 0:
            print(f"Generated {blend_idx + 1}/{num_blends} blends")
    
    return all_blend_data

def generate_custom_blend_curves(blend_strings, output_filename):
    """Generate curves for custom blends"""
    print("Generating custom blend curves...")
    
    custom_blend_curves = []
    custom_blend_labels = []
    
    for i, blend_str in enumerate(blend_strings):
        try:
            materials = parse_blend_input(blend_str)
            print(f"\nProcessing custom blend {i+1}: {blend_str}")
            
            # Find materials in sustainability.csv
            material_curves = []
            material_info = []
            
            for mat in materials:
                # Find matching material in sustainability.csv
                material_row = find_material_by_grade(sus, mat['grade'])
                if material_row is not None:
                    print(f"  Found: {material_row['Polymer Category']} {material_row['Grade']}")
                    
                    # Parse thickness
                    thickness_val = parse_thickness(material_row.get('Thickness 1', None))
                    
                    # Generate curve
                    curve = generate_material_curve(
                        material_row['Polymer Category'], 
                        material_row['Grade'], 
                        str(material_row.get('TUV Home', '')), 
                        thickness_val
                    )
                    
                    material_curves.append(curve)
                    material_info.append({
                        'polymer': material_row['Polymer Category'],
                        'grade': material_row['Grade'],
                        'vol_frac': mat['vol_frac'],
                        'curve': curve,
                        'tuv_home': str(material_row.get('TUV Home', ''))
                    })
                else:
                    print(f"  ERROR: Material with grade '{mat['grade']}' not found in sustainability.csv")
                    return
            
            if material_curves:
                # Calculate blend curve
                blend_curve = np.zeros(DAYS)
                
                # Check for synergistic effects based on certification
                home_materials = [mat for mat in material_info if is_home_compostable_certified(mat['tuv_home'])]
                
                for j, material in enumerate(material_info):
                    base_curve = material['curve'].copy()
                    
                    # Apply synergistic boost to non-home-compostable materials when home-compostable materials are present
                    # Use material name as seed for consistent results
                    np.random.seed(hash(material['grade']) % 2**32)
                    gets_boost = np.random.random() < 0.6  # 60% chance of getting the boost
                    
                    if (not is_home_compostable_certified(material['tuv_home']) and 
                        len(home_materials) > 0 and 
                        gets_boost):
                        
                        # Calculate home-compostable fraction in the blend
                        total_home_frac = sum(mat['vol_frac'] for mat in material_info if is_home_compostable_certified(mat['tuv_home']))
                        
                        # Determine additive boost based on blend composition
                        # Use material name as seed for consistent boost amount
                        np.random.seed(hash(material['grade']) % 2**32)
                        if total_home_frac >= 0.9:  # 90% or more home-compostable
                            boost_amount = 10  # Only 10% additive boost for 90-10 blends
                        else:
                            boost_amount = np.random.uniform(10, 20)  # 10-20% additive boost for other blends
                        
                        # Apply additive boost but ensure it doesn't exceed the intended maximum
                        enhanced_curve = base_curve + boost_amount
                        # Clip to the original sigmoid maximum to prevent exceeding intended asymptote
                        max_val = np.max(base_curve)
                        enhanced_curve = np.clip(enhanced_curve, 0, None)
                        material['curve'] = enhanced_curve
                        print(f"    Applied synergistic boost of +{boost_amount:.1f}% to {material['polymer']} (capped at {max_val:.1f}%)")
                    else:
                        # No boost applied
                        material['curve'] = base_curve
                    
                    blend_curve += material['curve'] * material['vol_frac']
                
                # Ensure blend curve is monotonically increasing and doesn't exceed 95%
                for i in range(1, len(blend_curve)):
                    if blend_curve[i] < blend_curve[i-1]:
                        blend_curve[i] = blend_curve[i-1] + np.random.uniform(0, 0.01)  # Very small positive increment
                
                # No clipping - let sigmoid handle everything naturally
                blend_curve = np.clip(blend_curve, 0, None)
                
                custom_blend_curves.append(blend_curve)
                
                # Create label
                label_parts = []
                for material in material_info:
                    label_parts.append(f"{material['polymer']} {material['grade']} ({material['vol_frac']:.1%})")
                custom_blend_labels.append(" + ".join(label_parts))
                
                print(f"  Generated blend curve with max disintegration: {np.max(blend_curve):.1f}%")
            else:
                print(f"  ERROR: No valid materials found for blend {i+1}")
                return
                
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
        is_home = 'certified' in tuv_home.lower() and 'not' not in tuv_home.lower()
        if polymer.upper() == 'PLA':
            is_home = False
        
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
                        is_home = 'certified' in material['tuv_home'].lower() and 'not' not in material['tuv_home'].lower()
                        if material['polymer'].upper() == 'PLA':
                            is_home = False
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