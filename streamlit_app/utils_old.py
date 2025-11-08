"""
Funciones utilitarias para generación de perfiles geotécnicos.
Este módulo provee funcionalidades para:
- Lectura y merge de archivos de cabeceras de sondeos y ensayos
- Cálculo de elevaciones de ensayos
- Ordenación de sondeos para eje X (por coordenada, PCA, etc.)
- Interpolación 2D en plano X-Z
- Enmascaramiento de zonas sin cobertura vertical
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, Rbf
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional, List, Union


def read_file(file_obj) -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y retorna un DataFrame.
    
    Parameters
    ----------
    file_obj : file-like object
        Objeto de archivo subido desde Streamlit.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos del archivo.
    """
    file_name = file_obj.name.lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(file_obj)
    elif file_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_obj)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {file_name}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas (elimina espacios extras, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a normalizar.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas normalizadas.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def merge_headers_and_samples(
    df_headers: pd.DataFrame,
    df_samples: pd.DataFrame,
    id_col_headers: str,
    id_col_samples: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Une los dataframes de cabeceras y ensayos por ID de sondeo.
    
    Parameters
    ----------
    df_headers : pd.DataFrame
        DataFrame con información de cabeceras (ID, x, y, cota).
    df_samples : pd.DataFrame
        DataFrame con ensayos (ID, profundidad, parámetros).
    id_col_headers : str
        Nombre de columna de ID en df_headers.
    id_col_samples : str
        Nombre de columna de ID en df_samples.
        
    Returns
    -------
    df_merged : pd.DataFrame
        DataFrame combinado.
    missing_ids : List[str]
        Lista de IDs en samples que no existen en headers.
    """
    # Identificar IDs sin match
    sample_ids = set(df_samples[id_col_samples].unique())
    header_ids = set(df_headers[id_col_headers].unique())
    missing_ids = list(sample_ids - header_ids)
    
    # Merge
    df_merged = df_samples.merge(
        df_headers,
        left_on=id_col_samples,
        right_on=id_col_headers,
        how='inner'
    )
    
    return df_merged, missing_ids


def calculate_z_param(
    df: pd.DataFrame,
    cota_col: str,
    profundidad_col: str,
    z_param_col: str = 'z_param'
) -> pd.DataFrame:
    """
    Calcula la elevación de cada ensayo: z_param = cota - profundidad_ensayo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos unidos.
    cota_col : str
        Nombre de columna con cota (elevación de cabeza).
    profundidad_col : str
        Nombre de columna con profundidad del ensayo.
    z_param_col : str
        Nombre de columna a crear para z_param.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columna z_param añadida.
    """
    df = df.copy()
    df[z_param_col] = df[cota_col] - df[profundidad_col]
    return df


def compute_borehole_bounds(
    df: pd.DataFrame,
    id_col: str,
    cota_col: str,
    profundidad_col: str
) -> pd.DataFrame:
    """
    Calcula los límites verticales (z_top, z_bottom) de cada sondeo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con ensayos.
    id_col : str
        Nombre de columna con ID de sondeo.
    cota_col : str
        Nombre de columna con cota.
    profundidad_col : str
        Nombre de columna con profundidad.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: sondeo, z_top, z_bottom, max_profundidad, n_ensayos.
    """
    # Agrupar por sondeo
    grouped = df.groupby(id_col).agg({
        cota_col: 'first',  # Cota es la misma para todos los ensayos del sondeo
        profundidad_col: 'max'  # Máxima profundidad
    }).reset_index()
    
    grouped['z_top'] = grouped[cota_col]
    grouped['z_bottom'] = grouped[cota_col] - grouped[profundidad_col]
    grouped['max_profundidad'] = grouped[profundidad_col]
    
    # Contar número de ensayos por sondeo
    n_ensayos = df.groupby(id_col).size().reset_index(name='n_ensayos')
    grouped = grouped.merge(n_ensayos, on=id_col)
    
    return grouped[[id_col, 'z_top', 'z_bottom', 'max_profundidad', 'n_ensayos']]


def order_boreholes_by_x(
    df_headers: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    method: str = 'x',
    manual_order: Optional[List[str]] = None
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Ordena los sondeos para el eje X del perfil según el método especificado.
    
    Parameters
    ----------
    df_headers : pd.DataFrame
        DataFrame con información de cabeceras.
    id_col : str
        Nombre de columna con ID.
    x_col : str
        Nombre de columna con coordenada X.
    y_col : str
        Nombre de columna con coordenada Y.
    method : str
        Método de ordenación:
        - 'x': Usar coordenada X real
        - 'xy_sort': Ordenar por X, luego por Y
        - 'pca': Proyección sobre eje principal (PCA)
        - 'manual': Orden manual proporcionado
    manual_order : Optional[List[str]]
        Lista de IDs en orden deseado (solo si method='manual').
        
    Returns
    -------
    borehole_positions : Dict[str, float]
        Diccionario mapeo sondeo_id -> posición_x.
    sorted_positions : np.ndarray
        Array de posiciones X ordenadas (únicas).
    """
    df = df_headers.copy()
    
    if method == 'x':
        # Usar coordenada X directamente
        df['x_pos'] = df[x_col]
        
    elif method == 'xy_sort':
        # Ordenar por X primero, luego por Y
        df = df.sort_values([x_col, y_col])
        # Asignar posiciones secuenciales
        df['x_pos'] = np.arange(len(df))
        
    elif method == 'pca':
        # Calcular eje principal (PCA) y proyectar
        coords = df[[x_col, y_col]].values
        pca = PCA(n_components=1)
        projections = pca.fit_transform(coords)
        df['x_pos'] = projections.flatten()
        
    elif method == 'manual':
        if manual_order is None:
            raise ValueError("manual_order debe proporcionarse cuando method='manual'")
        # Asignar posiciones según orden manual
        order_map = {id_val: i for i, id_val in enumerate(manual_order)}
        df['x_pos'] = df[id_col].map(order_map)
        # Eliminar sondeos no en la lista manual
        df = df.dropna(subset=['x_pos'])
        
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Crear diccionario de mapeo
    borehole_positions = dict(zip(df[id_col], df['x_pos']))
    sorted_positions = np.sort(df['x_pos'].unique())
    
    return borehole_positions, sorted_positions


def make_xz_grid(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    nx: int,
    nz: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una grilla 2D para el plano X-Z.
    
    Parameters
    ----------
    x_min, x_max : float
        Rango del eje X (posiciones de sondeos).
    z_min, z_max : float
        Rango del eje Z (elevaciones).
    nx, nz : int
        Número de puntos en X y Z.
        
    Returns
    -------
    grid_x, grid_z : np.ndarray
        Grillas 2D (meshgrid) para interpolación.
    """
    x = np.linspace(x_min, x_max, nx)
    z = np.linspace(z_max, z_min, nz)  # Z de arriba hacia abajo
    grid_x, grid_z = np.meshgrid(x, z)
    return grid_x, grid_z


def interpolate_xz_grid(
    points_xz: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    method: str = 'griddata',
    griddata_method: str = 'linear',
    rbf_func: str = 'multiquadric',
    idw_power: float = 2.0
) -> np.ndarray:
    """
    Interpola valores en una grilla X-Z.
    
    Parameters
    ----------
    points_xz : np.ndarray, shape (n, 2)
        Coordenadas (x_pos, z_param) de los puntos de ensayo.
    values : np.ndarray, shape (n,)
        Valores del parámetro en cada punto.
    grid_x, grid_z : np.ndarray
        Grillas 2D para interpolación.
    method : str
        Método de interpolación: 'griddata', 'rbf', 'idw'.
    griddata_method : str
        Método para griddata: 'linear', 'nearest', 'cubic'.
    rbf_func : str
        Función para RBF.
    idw_power : float
        Potencia para IDW.
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados en la grilla.
    """
    if method == 'griddata':
        grid_values = griddata(
            points_xz,
            values,
            (grid_x, grid_z),
            method=griddata_method
        )
        
    elif method == 'rbf':
        rbf = Rbf(
            points_xz[:, 0],
            points_xz[:, 1],
            values,
            function=rbf_func
        )
        grid_values = rbf(grid_x, grid_z)
        
    elif method == 'idw':
        grid_values = idw_interpolate_xz(
            points_xz,
            values,
            grid_x,
            grid_z,
            power=idw_power
        )
        
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    return grid_values


def idw_interpolate_xz(
    points_xz: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    power: float = 2.0
) -> np.ndarray:
    """
    Interpolación IDW (Inverse Distance Weighting) en grilla X-Z.
    
    Parameters
    ----------
    points_xz : np.ndarray, shape (n, 2)
        Coordenadas (x, z) de puntos de datos.
    values : np.ndarray, shape (n,)
        Valores en cada punto.
    grid_x, grid_z : np.ndarray
        Grillas 2D para interpolación.
    power : float
        Potencia para IDW.
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados.
    """
    # Aplanar grilla
    grid_points = np.c_[grid_x.ravel(), grid_z.ravel()]
    
    # Calcular distancias (vectorizado)
    distances = np.sqrt(
        np.sum((points_xz[None, :, :] - grid_points[:, None, :]) ** 2, axis=2)
    )
    
    # Evitar división por cero
    epsilon = 1e-10
    distances = np.maximum(distances, epsilon)
    
    # Calcular pesos
    weights = 1.0 / (distances ** power)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum
    
    # Interpolar
    grid_values = np.dot(weights, values)
    grid_values = grid_values.reshape(grid_x.shape)
    
    return grid_values


def create_vertical_mask(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    borehole_bounds: pd.DataFrame,
    borehole_positions: Dict[str, float],
    id_col: str,
    max_horizontal_distance: Optional[float] = None
) -> np.ndarray:
    """
    Crea una máscara para la grilla X-Z basada en cobertura vertical de sondeos.
    Las celdas fuera de la cobertura vertical real se marcan como NaN.
    
    Parameters
    ----------
    grid_x, grid_z : np.ndarray
        Grillas 2D.
    borehole_bounds : pd.DataFrame
        DataFrame con z_top, z_bottom por sondeo.
    borehole_positions : Dict[str, float]
        Mapeo sondeo_id -> x_pos.
    id_col : str
        Nombre de columna con ID de sondeo.
    max_horizontal_distance : Optional[float]
        Distancia horizontal máxima para considerar un sondeo relevante.
        Si None, se usa distancia al sondeo más cercano.
        
    Returns
    -------
    mask : np.ndarray
        Máscara booleana (True = válido, False = NaN).
    """
    mask = np.zeros_like(grid_x, dtype=bool)
    
    # Para cada columna X de la grilla
    for i in range(grid_x.shape[1]):
        x_col = grid_x[0, i]
        
        # Encontrar sondeos cercanos
        distances = []
        relevant_boreholes = []
        
        for _, row in borehole_bounds.iterrows():
            borehole_id = row[id_col]
            if borehole_id not in borehole_positions:
                continue
            
            x_borehole = borehole_positions[borehole_id]
            dist = abs(x_col - x_borehole)
            distances.append(dist)
            relevant_boreholes.append(row)
        
        if not relevant_boreholes:
            continue
        
        # Determinar distancia máxima
        if max_horizontal_distance is None:
            max_dist = min(distances) * 1.5  # 1.5x distancia al más cercano
        else:
            max_dist = max_horizontal_distance
        
        # Filtrar sondeos cercanos
        close_boreholes = [
            row for row, dist in zip(relevant_boreholes, distances)
            if dist <= max_dist
        ]
        
        if not close_boreholes:
            continue
        
        # Calcular envolvente vertical (unión de intervalos)
        z_top_max = max(row['z_top'] for row in close_boreholes)
        z_bottom_min = min(row['z_bottom'] for row in close_boreholes)
        
        # Marcar celdas válidas en esta columna
        for j in range(grid_z.shape[0]):
            z_cell = grid_z[j, i]
            if z_bottom_min <= z_cell <= z_top_max:
                mask[j, i] = True
    
    return mask


def apply_mask_to_grid(
    grid_values: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Aplica máscara a la grilla (marca como NaN las celdas inválidas).
    
    Parameters
    ----------
    grid_values : np.ndarray
        Valores interpolados.
    mask : np.ndarray
        Máscara booleana.
        
    Returns
    -------
    masked_grid : np.ndarray
        Grilla con máscara aplicada.
    """
    masked_grid = grid_values.copy()
    masked_grid[~mask] = np.nan
    return masked_grid


def get_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Obtiene lista de columnas numéricas de un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    exclude : List[str]
        Lista de columnas a excluir.
        
    Returns
    -------
    List[str]
        Lista de nombres de columnas numéricas.
    """
    if exclude is None:
        exclude = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def validate_merged_data(
    df: pd.DataFrame,
    required_cols: List[str]
) -> Tuple[bool, str]:
    """
    Valida que el DataFrame merged tenga las columnas requeridas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    required_cols : List[str]
        Lista de columnas requeridas.
        
    Returns
    -------
    is_valid : bool
        True si es válido.
    message : str
        Mensaje de error si no es válido.
    """
    if df.empty:
        return False, "El DataFrame está vacío tras el merge."
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Faltan columnas requeridas: {missing}"
    
    return True, "Validación exitosa"


def export_grid_to_dataframe(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    grid_values: np.ndarray
) -> pd.DataFrame:
    """
    Convierte grilla interpolada a DataFrame para exportar.
    
    Parameters
    ----------
    grid_x, grid_z : np.ndarray
        Grillas 2D.
    grid_values : np.ndarray
        Valores interpolados.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas X, Z, Value.
    """
    df = pd.DataFrame({
        'X': grid_x.ravel(),
        'Z': grid_z.ravel(),
        'Value': grid_values.ravel()
    })
    
    # Eliminar filas con NaN
    df = df.dropna(subset=['Value'])
    
    return df
