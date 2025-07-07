import pandas as pd
import numpy as np
import re
from scipy.stats import dirichlet

# --- GLOBAL SEED FOR REPRODUCIBILITY ---
GLOBAL_SEED = 42

def set_global_seed(seed=None):
    """Set global seed for all random operations"""
    if seed is None:
        seed = GLOBAL_SEED
    np.random.seed(seed)
    print(f"🌱 Set global seed to {seed} for reproducible results")

# Set initial seed
set_global_seed(GLOBAL_SEED)

# --- CONFIGURABLE PARAMETERS ---
DAYS = 200  # Number of days for the time series
SAMPLE_AREA = 100.0  # Default sample area (mm^2)
THICKNESS_DEFAULT = 1.0  # Default thickness (mm)

# --- LOAD DATA ---
sus = pd.read_csv('sustainability.csv')
sus.columns = [c.strip() for c in sus.columns]

# --- SIGMOID FUNCTION ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

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

def get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed=None):
    """Determine maximum disintegration percentage: TUV Home certification takes priority, then polymer type"""
    # Set seed for this specific material if provided
    if material_seed is not None:
        np.random.seed(material_seed)
    
    # First check TUV Home certification
    is_home_certified = is_home_compostable_certified(tuv_home)
    
    if is_home_certified:
        # Home-compostable certified materials: 90-95% max disintegration
        result = np.random.uniform(90, 95)
    else:
        # Not home-compostable certified - use polymer type classification
        if is_low_disintegration_polymer(polymer):
            result = np.random.uniform(0.5, 2)
        elif is_medium_disintegration_polymer(polymer):
            result = np.random.uniform(30, 80)
        else:
            result = np.random.uniform(10, 30)
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return result

def generate_material_curve(polymer, grade, tuv_home, thickness_val, days=DAYS, material_seed=None):
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
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Home-compostable certified: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}% - Thickness: {thickness_val:.3f}mm")
    else:
        # Not home-compostable certified: use hybrid classification
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        
        if max_disintegration < 5:  # Very low disintegration materials (petroleum-based)
            y = np.full_like(t, max_disintegration)
        else:
            k = 0.02
            t0 = 120
            y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Not home-compostable: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}%")
    
    # Set seed for noise generation
    if material_seed is not None:
        np.random.seed(material_seed + 1000)  # Offset for noise
    
    # Add very small noise for minimal realism (much reduced noise)
    y = y + np.random.normal(0, 0.1, size=y.shape)
    
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
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return y

def generate_material_curve_with_synergistic_boost(polymer, grade, tuv_home, thickness_val, home_fraction_in_blend, days=DAYS, material_seed=None):
    """Generate disintegration curve for a single material with synergistic boost from home-compostable materials"""
    t = np.arange(1, days+1)
    
    # Use ONLY the TUV Home certification data - NO HARDWIRED LOGIC
    is_home_compostable = is_home_compostable_certified(tuv_home)
    
    # Ensure thickness is valid
    if np.isnan(thickness_val) or thickness_val <= 0:
        thickness_val = THICKNESS_DEFAULT
    
    # Check if this material should get a synergistic boost
    should_get_boost = (
        not is_home_compostable and  # Not home-compostable certified
        home_fraction_in_blend > 0 and  # There are home-compostable materials in blend
        is_medium_disintegration_polymer(polymer)  # Only medium disintegrating polymers (removed PLA)
    )
    
    if is_home_compostable:
        # Home-compostable certified: sigmoid curve with high disintegration
        base_k = 0.08
        base_t0 = 70
        
        # Reduced thickness effect for very thin materials
        thickness_factor = (thickness_val / 1.0) ** 0.1  # Much smaller thickness effect
        k = base_k * thickness_factor
        t0 = base_t0 / thickness_factor
        
        # Use hybrid maximum disintegration (certification takes priority)
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Home-compostable certified: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}% - Thickness: {thickness_val:.3f}mm")
    else:
        # Not home-compostable certified: use hybrid classification
        max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        
        if max_disintegration < 5:  # Very low disintegration materials (petroleum-based)
            y = np.full_like(t, max_disintegration)
        else:
            k = 0.02
            t0 = 120
            
            # Apply synergistic boost if eligible
            if should_get_boost:
                # Use home-compostable kinetics but with proportional max disintegration boost
                base_k = 0.08
                base_t0 = 70
                
                # Apply thickness effect like home-compostable materials
                thickness_factor = (thickness_val / 1.0) ** 0.1
                k = base_k * thickness_factor
                t0 = base_t0 / thickness_factor
                
                # Proportional max disintegration boost based on home-compostable fraction
                max_boost_percent = 20  # Maximum 20% boost
                actual_boost = max_boost_percent * home_fraction_in_blend  # Proportional to home fraction
                max_disintegration = min(max_disintegration + actual_boost, 95)  # Cap at 95%
                
                print(f"    Synergistic boost applied to {polymer} {grade}: +{actual_boost:.1f}% max ({home_fraction_in_blend:.1%} of max), using home-compostable kinetics")
            
            y = sigmoid(t, max_disintegration, k, t0)
        
        print(f"    Not home-compostable: {polymer} {grade} - Max disintegration: {max_disintegration:.1f}%")
    
    # Set seed for noise generation
    if material_seed is not None:
        np.random.seed(material_seed + 2000)  # Offset for synergistic noise
    
    # Add very small noise for minimal realism (much reduced noise)
    y = y + np.random.normal(0, 0.1, size=y.shape)
    
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
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
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