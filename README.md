# Profiles
Perfiles de parametros

## Geotechnical Contours Tool

A Streamlit application for visualizing geotechnical parameters using contour plots based on borehole data.

### Features

- Load data from CSV or Excel files
- Interactive column mapping interface
- Multiple interpolation methods:
  - Linear griddata
  - Nearest neighbor griddata
  - Cubic griddata
  - Radial Basis Function (RBF)
  - Inverse Distance Weighting (IDW)
- Support for single or multiple elevation slices
- Customizable visualization (resolution, colormap, contour levels)
- Export plots as PNG images

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit application:

```bash
cd contornos_app
streamlit run contornos.py
```

### Data Format

Your CSV or Excel file should contain columns for:
- Sondeo (Borehole ID)
- x (Easting coordinate)
- y (Northing coordinate)
- cota (Head elevation)
- profundidad (Maximum depth from head)
- Parameter to interpolate (e.g., SPT, soil type, etc.)

The application will guide you through mapping your file's columns to these required fields.
