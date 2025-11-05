import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from utils import read_table, make_grid, interpolate_grid, idw_interpolate, compute_z_bounds

st.set_page_config(page_title="Contornos Geotécnicos", layout="wide")

st.title("Herramienta de contornos geotécnicos")

st.markdown(
    """
Carga un archivo CSV o Excel con mediciones por sondeo. Selecciona las columnas que corresponden a:
- Sondeo (ID)
- x (coordenada Easting)
- y (coordenada Northing)
- cota (elevación de cabeza)
- profundidad (profundidad máxima desde la cota)
- parámetro (valor a interpolar)
"""
)

uploaded = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xls", "xlsx"])
if uploaded is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()

# Leer archivo
try:
    df = read_table(uploaded)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

st.subheader("Vista previa de datos")
st.dataframe(df.head())

cols = df.columns.tolist()

with st.form("mapeo_columnas"):
    st.write("Mapea las columnas del archivo a las variables requeridas")
    col_sondeo = st.selectbox("Columna: Sondeo (ID)", options=cols, index=0)
    col_x = st.selectbox("Columna: x (coordenada)", options=cols, index=min(1, len(cols)-1))
    col_y = st.selectbox("Columna: y (coordenada)", options=cols, index=min(2, len(cols)-1))
    col_cota = st.selectbox("Columna: cota (elevación cabeza)", options=cols, index=min(3, len(cols)-1))
    col_profundidad = st.selectbox("Columna: profundidad (profundidad máxima)", options=cols, index=min(4, len(cols)-1))
    col_param = st.selectbox("Columna: parámetro a graficar", options=cols, index=min(5, len(cols)-1))
    submitted = st.form_submit_button("Aplicar mapeo")

if not submitted:
    st.stop()

# Prepare dataframe
df_clean = df[[col_sondeo, col_x, col_y, col_cota, col_profundidad, col_param]].copy()
df_clean.columns = ["sondeo", "x", "y", "cota", "profundidad", "param"]
# drop rows with missing coords or param
df_clean = df_clean.dropna(subset=["x", "y", "cota", "profundidad", "param"])
# ensure numeric
for c in ["x", "y", "cota", "profundidad", "param"]:
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
df_clean = df_clean.dropna(subset=["x", "y", "cota", "profundidad", "param"])

if df_clean.empty:
    st.error("No quedan filas válidas tras limpieza. Revisa el archivo.")
    st.stop()

# compute top and bottom elevations
df_clean["z_top"] = df_clean["cota"]
df_clean["z_bottom"] = df_clean["cota"] - df_clean["profundidad"]

z_min, z_max = compute_z_bounds(df_clean)
st.write(f"Rango de elevaciones disponible: z máximo (cabezas) = {z_max:.2f}, z mínimo (fondos) = {z_min:.2f}")

# UI for interpolation and plotting
st.sidebar.subheader("Configuración de interpolación")
method = st.sidebar.selectbox("Método de interpolación", options=["griddata_linear", "griddata_nearest", "griddata_cubic", "rbf", "idw"])
grid_nx = st.sidebar.number_input("Resolución grilla en X (puntos)", min_value=50, max_value=1000, value=200, step=10)
grid_ny = st.sidebar.number_input("Resolución grilla en Y (puntos)", min_value=50, max_value=1000, value=200, step=10)
levels = st.sidebar.number_input("Número de niveles de contorno", min_value=3, max_value=50, value=10)
# Cache the colormap list to improve performance
@st.cache_data
def get_colormap_list():
    return sorted([m for m in plt.colormaps()])

colormap_list = get_colormap_list()
cmap = st.sidebar.selectbox("Colormap", options=colormap_list, index=colormap_list.index("viridis") if "viridis" in colormap_list else 0)

st.sidebar.markdown("Seleccione elevación (z) para el plano de contorno")
z_mode = st.sidebar.radio("Modo", options=["Un nivel", "Múltiples niveles"])
if z_mode == "Un nivel":
    z_sel = st.sidebar.slider("Elevación z (unidad misma que cota/profundidad)", min_value=float(z_min), max_value=float(z_max), value=float((z_min + z_max) / 2))
    z_levels = [z_sel]
else:
    n_slices = st.sidebar.number_input("Número de planos equiespaciados", min_value=2, max_value=10, value=3)
    z_levels = list(np.linspace(z_max, z_min, n_slices))

st.sidebar.markdown("Opciones adicionales")
idw_power = st.sidebar.slider("IDW power (si IDW seleccionado)", min_value=0.5, max_value=4.0, value=2.0)
rbf_func = st.sidebar.selectbox("RBF function", options=["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic"], index=0)

plot_button = st.button("Generar contorno(s)")

if not plot_button:
    st.info("Presiona 'Generar contorno(s)' para ejecutar la interpolación y mostrar la/s figura/s.")
    st.stop()

# For each z_level, filter points whose interval covers z
plots = []
for z in z_levels:
    # select points where z is between z_bottom and z_top
    mask = (df_clean["z_bottom"] <= z) & (z <= df_clean["z_top"])
    pts = df_clean[mask]
    st.write(f"Elevación z = {z:.2f} → puntos que cubren el plano: {len(pts)}")
    if len(pts) < 3:
        st.warning(f"Pocos puntos ({len(pts)}) cubren el plano z={z:.2f}; la interpolación no es posible. Se requieren al menos 3 puntos.")
        continue
    # create grid
    xmin, xmax = df_clean["x"].min(), df_clean["x"].max()
    ymin, ymax = df_clean["y"].min(), df_clean["y"].max()
    grid_x, grid_y = make_grid(xmin, xmax, ymin, ymax, grid_nx, grid_ny)
    # interpolate
    try:
        if method.startswith("griddata"):
            m = method.split("_")[1]
            grid_z = interpolate_grid(pts[["x", "y"]].values, pts["param"].values, grid_x, grid_y, method="griddata", griddata_method=m)
        elif method == "rbf":
            grid_z = interpolate_grid(pts[["x", "y"]].values, pts["param"].values, grid_x, grid_y, method="rbf", rbf_func=rbf_func)
        elif method == "idw":
            grid_z = idw_interpolate(pts[["x", "y"]].values, pts["param"].values, grid_x, grid_y, power=idw_power)
        else:
            st.error("Método desconocido")
            continue
    except Exception as e:
        st.error(f"Error durante interpolación para z={z}: {e}")
        continue

    # Check if grid has sufficient finite values for plotting
    finite_count = np.sum(np.isfinite(grid_z))
    if finite_count < 3:
        st.warning(f"Interpolación para z={z:.2f} produjo muy pocos valores finitos ({finite_count}). No se puede graficar.")
        continue
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # mask nan
    im = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)
    ax.scatter(pts["x"], pts["y"], c="k", s=15, label="Puntos usados")
    ax.set_title(f"Contorno de '{col_param}' a z = {z:.2f} ({method})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(col_param)
    st.pyplot(fig)
    # keep for export if needed
    plots.append((z, fig))

# Export: allow download of last figure as PNG
if plots:
    last_fig = plots[-1][1]
    buf = BytesIO()
    last_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Descargar última figura (PNG)", data=buf, file_name="contorno.png", mime="image/png")
