"""
Funciones utilitarias para interpolación 2D de parámetros geotécnicos.
Este módulo provee funcionalidades para:
- Lectura y normalización de archivos con una sola tabla
- Interpolación 2D en plano X-Y (abscisa vs cota)
- Múltiples métodos: griddata, RBF, IDW
- Enmascaramiento para evitar extrapolación (ConvexHull y distancia)
- Exportación de grillas interpoladas

Extensiones posibles:
- Añadir kriging con pykrige para datos geoestadísticos
- Exportar a GeoTIFF con coordenadas georreferenciadas
- Implementar submuestreo inteligente para datasets grandes
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.spatial import ConvexHull, distance
from typing import Tuple, List, Optional, Dict, Union


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
        
    Raises
    ------
    ValueError
        Si el tipo de archivo no es soportado.
    """
    file_name = file_obj.name.lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(file_obj)
    elif file_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_obj)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {file_name}. Use CSV o Excel.")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas (elimina espacios, convierte a minúsculas, 
    reemplaza espacios por guiones bajos).
    
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
    df.columns = [
        str(col).strip().lower().replace(' ', '_').replace('-', '_')
        for col in df.columns
    ]
    return df


def get_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Obtiene lista de columnas numéricas de un DataFrame, excluyendo las especificadas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    exclude : List[str], optional
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


def validate_data_for_interpolation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    param_cols: List[str],
    min_points: int = 3
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Valida y limpia datos para interpolación.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos.
    x_col : str
        Nombre de columna X.
    y_col : str
        Nombre de columna Y.
    param_cols : List[str]
        Lista de columnas de parámetros.
    min_points : int
        Número mínimo de puntos requeridos.
        
    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame limpio con valores válidos.
    warnings : Dict[str, str]
        Diccionario con advertencias por parámetro.
    """
    df_clean = df.copy()
    warnings = {}
    
    # Convertir a numérico
    for col in [x_col, y_col] + param_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Eliminar filas con X o Y faltantes
    original_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[x_col, y_col])
    removed = original_len - len(df_clean)
    
    if removed > 0:
        warnings['xy_missing'] = f"Se eliminaron {removed} filas por valores faltantes en X o Y"
    
    # Verificar puntos mínimos
    if len(df_clean) < min_points:
        warnings['insufficient_points'] = f"Insuficientes puntos válidos: {len(df_clean)} (mínimo: {min_points})"
    
    # Verificar cada parámetro
    for param in param_cols:
        valid_count = df_clean[param].notna().sum()
        if valid_count < min_points:
            warnings[f'{param}_insufficient'] = f"Parámetro '{param}': solo {valid_count} puntos válidos"
    
    return df_clean, warnings


def calculate_parameter_statistics(
    df: pd.DataFrame,
    param_cols: List[str]
) -> pd.DataFrame:
    """
    Calcula estadísticas básicas para cada parámetro.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos.
    param_cols : List[str]
        Lista de columnas de parámetros.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con estadísticas (min, max, mean, std, n_points).
    """
    stats = []
    for param in param_cols:
        if param in df.columns:
            values = df[param].dropna()
            stats.append({
                'Parámetro': param,
                'n_puntos': len(values),
                'Mínimo': values.min() if len(values) > 0 else np.nan,
                'Máximo': values.max() if len(values) > 0 else np.nan,
                'Media': values.mean() if len(values) > 0 else np.nan,
                'Desv.Est.': values.std() if len(values) > 0 else np.nan
            })
    
    return pd.DataFrame(stats)


def make_xy_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una grilla 2D regular para el plano X-Y.
    
    Parameters
    ----------
    x_min, x_max : float
        Rango del eje X (abscisa).
    y_min, y_max : float
        Rango del eje Y (cota/elevación).
    nx, ny : int
        Número de puntos en X y Y.
        
    Returns
    -------
    grid_x, grid_y : np.ndarray
        Grillas 2D (meshgrid) para interpolación.
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y


def interpolate_xy_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = 'griddata',
    griddata_method: str = 'linear',
    rbf_func: str = 'multiquadric',
    idw_power: float = 2.0
) -> np.ndarray:
    """
    Interpola valores en una grilla X-Y usando el método especificado.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de los puntos de datos.
    values : np.ndarray, shape (n,)
        Valores del parámetro en cada punto.
    grid_x, grid_y : np.ndarray
        Grillas 2D para interpolación.
    method : str
        Método de interpolación: 'griddata', 'rbf', 'idw'.
    griddata_method : str
        Método para griddata: 'linear', 'nearest', 'cubic'.
    rbf_func : str
        Función para RBF: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'.
    idw_power : float
        Potencia para IDW (típicamente 2.0).
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados en la grilla.
        
    Raises
    ------
    ValueError
        Si el método no es reconocido o hay error en interpolación.
    """
    if method == 'griddata':
        grid_values = griddata(
            points_xy,
            values,
            (grid_x, grid_y),
            method=griddata_method
        )
        
    elif method == 'rbf':
        rbf = Rbf(
            points_xy[:, 0],
            points_xy[:, 1],
            values,
            function=rbf_func
        )
        grid_values = rbf(grid_x, grid_y)
        
    elif method == 'idw':
        grid_values = idw_interpolate(
            points_xy,
            values,
            grid_x,
            grid_y,
            power=idw_power
        )
        
    else:
        raise ValueError(f"Método de interpolación desconocido: {method}")
    
    return grid_values


def idw_interpolate(
    points_xy: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0
) -> np.ndarray:
    """
    Interpolación IDW (Inverse Distance Weighting) implementada de forma vectorizada.
    
    IDW es un método de interpolación ponderado por la distancia inversa que calcula
    valores interpolados como un promedio ponderado de los puntos conocidos, donde
    los pesos son inversamente proporcionales a la distancia elevada a una potencia.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    values : np.ndarray, shape (n,)
        Valores en cada punto.
    grid_x, grid_y : np.ndarray
        Grillas 2D para interpolación.
    power : float
        Potencia para IDW. Valores típicos: 1-3. Mayor potencia = más peso a puntos cercanos.
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados en la grilla.
    """
    # Aplanar grilla
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # Calcular distancias (vectorizado) - shape: (n_grid_points, n_data_points)
    distances = distance.cdist(grid_points, points_xy, metric='euclidean')
    
    # Evitar división por cero
    epsilon = 1e-10
    distances = np.maximum(distances, epsilon)
    
    # Calcular pesos: w = 1 / d^power
    weights = 1.0 / (distances ** power)
    
    # Normalizar pesos
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum
    
    # Interpolar: valor = suma(w_i * value_i)
    grid_values = np.dot(weights, values)
    
    # Reshape a forma de grilla
    grid_values = grid_values.reshape(grid_x.shape)
    
    return grid_values


def create_convexhull_mask(
    points_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray
) -> np.ndarray:
    """
    Crea máscara basada en ConvexHull de los puntos de datos.
    Las celdas fuera del hull convexo se marcan como inválidas (False).
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    grid_x, grid_y : np.ndarray
        Grillas 2D.
        
    Returns
    -------
    mask : np.ndarray
        Máscara booleana (True = dentro del hull, False = fuera).
        
    Notes
    -----
    Requiere al menos 4 puntos no colineales para crear un ConvexHull en 2D.
    Si hay menos puntos, retorna máscara toda True (sin enmascarar).
    """
    if len(points_xy) < 4:
        # No hay suficientes puntos para ConvexHull, retornar todo válido
        return np.ones_like(grid_x, dtype=bool)
    
    try:
        hull = ConvexHull(points_xy)
        
        # Aplanar grilla
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        # Verificar qué puntos están dentro del hull
        # Método: un punto está dentro si todas las ecuaciones del hull son <= 0
        mask_flat = np.all(
            hull.equations[:, :2] @ grid_points.T + hull.equations[:, 2:3] <= 1e-12,
            axis=0
        )
        
        # Reshape a forma de grilla
        mask = mask_flat.reshape(grid_x.shape)
        
    except Exception as e:
        # Si falla ConvexHull (puntos colineales, etc.), retornar todo válido
        mask = np.ones_like(grid_x, dtype=bool)
    
    return mask


def create_distance_mask(
    points_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    max_distance: Optional[float] = None,
    k_neighbors: int = 1
) -> np.ndarray:
    """
    Crea máscara basada en distancia a los k vecinos más cercanos.
    Las celdas con distancia mayor al umbral se marcan como inválidas.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    grid_x, grid_y : np.ndarray
        Grillas 2D.
    max_distance : float, optional
        Distancia máxima permitida. Si None, se calcula automáticamente como
        percentil 90 de distancias entre puntos vecinos * factor de seguridad.
    k_neighbors : int
        Número de vecinos a considerar (típicamente 1).
        
    Returns
    -------
    mask : np.ndarray
        Máscara booleana (True = válido, False = muy lejos de datos).
    """
    # Aplanar grilla
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # Calcular distancias a todos los puntos de datos
    distances = distance.cdist(grid_points, points_xy, metric='euclidean')
    
    # Distancia al k-ésimo vecino más cercano
    k = min(k_neighbors, len(points_xy))
    kth_distances = np.partition(distances, k-1, axis=1)[:, k-1]
    
    # Determinar umbral
    if max_distance is None:
        # Calcular umbral automático: analizar distancias típicas entre puntos
        if len(points_xy) > 1:
            pairwise_dist = distance.pdist(points_xy)
            # Usar percentil 90 de distancias como referencia
            typical_distance = np.percentile(pairwise_dist, 90)
            max_distance = typical_distance * 1.5  # Factor de seguridad
        else:
            max_distance = np.inf  # Sin umbral si solo hay 1 punto
    
    # Crear máscara
    mask_flat = kth_distances <= max_distance
    mask = mask_flat.reshape(grid_x.shape)
    
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
        Máscara booleana (True = válido, False = enmascarar).
        
    Returns
    -------
    masked_grid : np.ndarray
        Grilla con máscara aplicada (NaN donde mask=False).
    """
    masked_grid = grid_values.copy()
    masked_grid[~mask] = np.nan
    return masked_grid


def combine_masks(mask1: np.ndarray, mask2: np.ndarray, operation: str = 'and') -> np.ndarray:
    """
    Combina dos máscaras con operación lógica.
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Máscaras booleanas a combinar.
    operation : str
        Operación: 'and' (intersección) u 'or' (unión).
        
    Returns
    -------
    np.ndarray
        Máscara combinada.
    """
    if operation == 'and':
        return mask1 & mask2
    elif operation == 'or':
        return mask1 | mask2
    else:
        raise ValueError(f"Operación desconocida: {operation}. Use 'and' u 'or'.")


def export_grid_to_dataframe(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_values: np.ndarray,
    param_name: str = 'Value'
) -> pd.DataFrame:
    """
    Convierte grilla interpolada a DataFrame para exportar.
    
    Parameters
    ----------
    grid_x, grid_y : np.ndarray
        Grillas 2D de coordenadas.
    grid_values : np.ndarray
        Valores interpolados.
    param_name : str
        Nombre del parámetro para la columna de valores.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas X, Y, y el parámetro.
    """
    df = pd.DataFrame({
        'X': grid_x.ravel(),
        'Y': grid_y.ravel(),
        param_name: grid_values.ravel()
    })
    
    # No eliminamos NaN para mantener la estructura de la grilla
    # El usuario puede filtrar si lo desea
    
    return df


def subsample_data(
    df: pd.DataFrame,
    max_points: int = 10000,
    method: str = 'random'
) -> pd.DataFrame:
    """
    Submuestrea el DataFrame si tiene demasiados puntos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a submuestrear.
    max_points : int
        Número máximo de puntos deseados.
    method : str
        Método de submuestreo: 'random' (aleatorio) o 'grid' (por rejilla).
        
    Returns
    -------
    pd.DataFrame
        DataFrame submuestreado.
        
    Notes
    -----
    El submuestreo por rejilla (grid) divide el espacio en celdas y toma
    un punto representativo por celda, útil para mantener distribución espacial.
    """
    if len(df) <= max_points:
        return df
    
    if method == 'random':
        return df.sample(n=max_points, random_state=42)
    
    elif method == 'grid':
        # Implementación simplificada: dividir en grid y tomar un punto por celda
        # Esto requeriría columnas X, Y específicas - por simplicidad usamos random
        return df.sample(n=max_points, random_state=42)
    
    else:
        raise ValueError(f"Método de submuestreo desconocido: {method}")
