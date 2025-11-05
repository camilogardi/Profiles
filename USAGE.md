# Geotechnical Contours Tool - Usage Guide

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

Navigate to the application directory and launch Streamlit:

```bash
cd contornos_app
streamlit run contornos.py
```

The application will open in your web browser at `http://localhost:8501`

### 3. Using Sample Data

A sample data file (`sample_data.csv`) is provided for testing. Upload this file to see the application in action.

## Data Format Requirements

Your CSV or Excel file must contain at least the following columns (names can be different, you'll map them in the UI):

- **Sondeo (ID)**: Unique identifier for each borehole
- **x**: Easting coordinate
- **y**: Northing coordinate  
- **cota**: Head elevation (surface elevation)
- **profundidad**: Maximum depth from the head elevation
- **parameter**: The value to interpolate (e.g., SPT, soil density, moisture content)

### Example Data Format

```csv
sondeo,x,y,cota,profundidad,SPT
S1,100,200,50,15,25
S2,150,200,52,18,30
S3,200,200,51,16,28
```

## Workflow

### Step 1: Upload Data
- Click "Browse files" and select your CSV or Excel file
- The application will display a preview of your data

### Step 2: Map Columns
- Use the dropdown menus to map your file's columns to the required variables
- Click "Aplicar mapeo" to confirm

### Step 3: Configure Interpolation
Use the sidebar to configure:

**Interpolation Method:**
- `griddata_linear`: Linear interpolation (fast, smooth)
- `griddata_nearest`: Nearest neighbor (preserves discrete values)
- `griddata_cubic`: Cubic interpolation (very smooth, can overshoot)
- `rbf`: Radial Basis Function (smooth, good for scattered data)
- `idw`: Inverse Distance Weighting (weighted average)

**Grid Resolution:**
- Set the number of grid points in X and Y directions (higher = more detail, slower)

**Contour Levels:**
- Number of contour lines to display (3-50)

**Colormap:**
- Choose from 180+ matplotlib colormaps

### Step 4: Select Elevation Slice(s)
- **Un nivel**: Single horizontal slice at a specific elevation
- **Múltiples niveles**: Multiple equally-spaced slices

### Step 5: Generate Contours
- Click "Generar contorno(s)" to run the interpolation and display results
- Each plot shows:
  - Contour map of the interpolated parameter
  - Black points indicating boreholes that cover that elevation
  - Colorbar showing parameter values

### Step 6: Export
- Download the last generated figure as PNG using the download button

## Interpolation Methods Explained

### Griddata (Linear, Nearest, Cubic)
- Based on Delaunay triangulation
- **Linear**: Fast, good for most cases
- **Nearest**: Use when you want discrete zones
- **Cubic**: Smoothest, may create artifacts outside data range

### RBF (Radial Basis Function)
- Creates smooth surfaces
- Functions: multiquadric, inverse, gaussian, linear, cubic, quintic
- Good for scattered data with smooth variations

### IDW (Inverse Distance Weighting)
- Weighted average based on distance
- Power parameter controls influence (higher = more local)
- Simple and robust, no overshoot

## Tips

1. **Not Enough Points**: If you get warnings about few points covering a slice, try:
   - Adjusting the elevation level
   - Using fewer slices
   - Checking your depth data

2. **Performance**: 
   - Lower grid resolution (50-100) for quick previews
   - Higher resolution (200-500) for final figures

3. **Artifacts**:
   - Use `griddata_nearest` or `idw` to avoid overshooting
   - Try different RBF functions
   - Increase IDW power for more localized interpolation

4. **Missing Data**:
   - The application automatically removes rows with missing coordinates or parameters
   - Check the cleaned data count after column mapping

## Troubleshooting

**Error: "Pocos puntos cubren el plano"**
- The selected elevation slice doesn't intersect enough boreholes
- Solution: Choose a different elevation or use "Múltiples niveles" to see which elevations have data

**Blank Contour Plot**
- May occur with cubic interpolation when data is sparse
- Solution: Try linear or nearest interpolation

**Out of Memory**
- Grid resolution is too high
- Solution: Reduce grid_nx and grid_ny values

**Import Errors**
- Missing dependencies
- Solution: `pip install -r requirements.txt`
