# Polymer Blend Disintegration Generator

A Streamlit web application for generating disintegration curves for custom polymer blends using physics-based modeling at 28°C.

## 🚀 Live App

Access the live application: [Coming Soon - Deploy to Streamlit Cloud]

## Features

- **Material Selection**: Choose from 100+ polymer grades from the sustainability database
- **Blend Generation**: Create custom blends with up to 5 materials
- **Visual Analysis**: Interactive plots with max disintegration and 90-day labels
- **Synergistic Effects**: Model realistic interactions between home and non-home compostable materials
- **Certification Data**: Based on TUV Home certification standards

## Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amerelsamman/Home-Compostability.git
   cd Home-Compostability
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app locally**:
   ```bash
   streamlit run augment_blends_28C_cli.py
   ```

4. **Access the app**: Open http://localhost:8501 in your browser

## Usage

1. **Select Materials**: Choose up to 5 polymer grades from the dropdown
2. **Set Volume Fractions**: Ensure they sum to 1.0 (100%)
3. **Generate**: Click the button to see the blend curve
4. **Analyze**: View max disintegration, 90-day values, and material properties

## Model Features

- **Physics-based**: Uses sigmoid curves with realistic kinetics
- **Certification-aware**: Respects TUV Home certification data
- **Thickness effects**: Accounts for material thickness in disintegration
- **Synergistic boosts**: Models interactions between different polymer types
- **Reproducible**: Consistent results with deterministic seeding

## Data Sources

- **sustainability.csv**: Contains polymer grades, TUV Home certification, and thickness data
- **Model parameters**: Based on literature and experimental data for 28°C composting

## Deployment

This app is designed to be deployed on Streamlit Cloud for easy sharing with teammates.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `streamlit run augment_blends_28C_cli.py`
5. Submit a pull request

## License

This project is for research and educational purposes. 