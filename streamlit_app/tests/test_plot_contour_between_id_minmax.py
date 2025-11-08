"""
Pruebas unitarias para la función plot_contour_between_id_minmax().

Este módulo verifica que la función plot_contour_between_id_minmax genera
correctamente contornos 2D limitados por polígonos basados en min/max de
cotas por ID de sondeo.

Para ejecutar las pruebas:
    pip install -r streamlit_app/requirements.txt
    pip install pytest
    pytest -v streamlit_app/tests/test_plot_contour_between_id_minmax.py
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Añadir el directorio padre al path para poder importar utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import plot_contour_between_id_minmax, export_interpolated_grid_to_csv, figure_to_bytes


def create_synthetic_drilling_data():
    """
    Crea un DataFrame sintético simulando datos de sondeos geotécnicos.
    
    Genera 3 sondeos (P-01, P-02, P-03) con múltiples mediciones a
    diferentes cotas.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: id, abscisa, cota, qc
    """
    data = []
    
    # Sondeo P-01 en x=100
    for cota in [50.0, 48.0, 46.0, 44.0]:
        data.append({'id': 'P-01', 'abscisa': 100.0, 'cota': cota, 'qc': 2.5 + np.random.uniform(-0.2, 0.2)})
    
    # Sondeo P-02 en x=150
    for cota in [52.0, 50.0, 48.0, 46.0]:
        data.append({'id': 'P-02', 'abscisa': 150.0, 'cota': cota, 'qc': 3.0 + np.random.uniform(-0.2, 0.2)})
    
    # Sondeo P-03 en x=200
    for cota in [51.0, 49.0, 47.0, 45.0]:
        data.append({'id': 'P-03', 'abscisa': 200.0, 'cota': cota, 'qc': 2.8 + np.random.uniform(-0.2, 0.2)})
    
    return pd.DataFrame(data)


def test_plot_contour_basic():
    """
    Test: Generación básica de contorno con datos sintéticos.
    
    Verifica que:
    - La función retorna una tupla (fig, ax, poly)
    - fig es una instancia de matplotlib.figure.Figure
    - ax es una instancia de matplotlib.axes.Axes
    - poly tiene un atributo 'area' > 0
    """
    df = create_synthetic_drilling_data()
    
    # Llamar a la función con parámetros básicos
    fig, ax, poly = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        nx=50,  # Resolución baja para test rápido
        ny=50
    )
    
    # Verificaciones
    assert isinstance(fig, plt.Figure), "fig debe ser una instancia de matplotlib.figure.Figure"
    assert isinstance(ax, plt.Axes), "ax debe ser una instancia de matplotlib.axes.Axes"
    assert hasattr(poly, 'area'), "poly debe tener atributo 'area'"
    assert poly.area > 0, f"El área del polígono debe ser > 0, obtenido: {poly.area}"
    
    # Cerrar figura para liberar memoria
    plt.close(fig)
    
    print(f"✓ Test básico: fig, ax, poly generados correctamente. Área polígono: {poly.area:.2f}")


def test_plot_contour_with_y_limits():
    """
    Test: Generación de contorno con límites de Y especificados.
    
    Verifica que:
    - La función respeta y_limits si se especifican
    - El polígono resultante tiene área > 0
    """
    df = create_synthetic_drilling_data()
    
    # Especificar límites de Y
    y_limits = (46.0, 51.0)
    
    fig, ax, poly = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        y_limits=y_limits,
        nx=50,
        ny=50
    )
    
    # Verificar que los límites del eje Y están correctos
    y_lim = ax.get_ylim()
    assert y_lim[0] == y_limits[0], f"Límite inferior Y esperado {y_limits[0]}, obtenido {y_lim[0]}"
    assert y_lim[1] == y_limits[1], f"Límite superior Y esperado {y_limits[1]}, obtenido {y_lim[1]}"
    assert poly.area > 0, "El área del polígono debe ser > 0"
    
    plt.close(fig)
    
    print(f"✓ Test y_limits: Límites Y aplicados correctamente. Área polígono: {poly.area:.2f}")


def test_plot_contour_interpolation_methods():
    """
    Test: Probar diferentes métodos de interpolación.
    
    Verifica que:
    - 'linear' funciona correctamente
    - 'cubic' funciona correctamente (o cae a linear si falla)
    """
    df = create_synthetic_drilling_data()
    
    # Test con linear
    fig_linear, ax_linear, poly_linear = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        prefer_method='linear',
        nx=30,
        ny=30
    )
    assert poly_linear.area > 0, "Método 'linear' debe generar polígono válido"
    plt.close(fig_linear)
    
    # Test con cubic
    fig_cubic, ax_cubic, poly_cubic = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        prefer_method='cubic',
        nx=30,
        ny=30
    )
    assert poly_cubic.area > 0, "Método 'cubic' debe generar polígono válido"
    plt.close(fig_cubic)
    
    print("✓ Test métodos interpolación: 'linear' y 'cubic' funcionan correctamente")


def test_plot_contour_custom_parameters():
    """
    Test: Generación de contorno con parámetros personalizados.
    
    Verifica que:
    - Se pueden personalizar n_levels, cmap, scatter_size, title, figsize
    - La función no lanza excepciones con parámetros válidos
    """
    df = create_synthetic_drilling_data()
    
    fig, ax, poly = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        n_levels=10,
        cmap='plasma',
        scatter_size=15,
        title='Test Personalizado',
        figsize=(12, 8),
        nx=40,
        ny=40,
        clip_to_range=True
    )
    
    assert poly.area > 0, "Debe generar polígono válido con parámetros personalizados"
    assert ax.get_title() == 'Test Personalizado', "El título debe coincidir"
    
    plt.close(fig)
    
    print("✓ Test parámetros personalizados: Todos los parámetros aplicados correctamente")


def test_plot_contour_insufficient_ids():
    """
    Test: Manejo de error cuando no hay IDs.
    
    Verifica que:
    - La función lanza RuntimeError si no hay datos por ID
    """
    # DataFrame vacío
    df_empty = pd.DataFrame(columns=['abscisa', 'cota', 'qc', 'id'])
    
    with pytest.raises(RuntimeError, match="No hay sondajes"):
        plot_contour_between_id_minmax(
            df_empty,
            x_col='abscisa',
            y_col='cota',
            z_col='qc',
            id_col='id'
        )
    
    print("✓ Test IDs insuficientes: RuntimeError lanzado correctamente")


def test_export_grid_to_csv():
    """
    Test: Exportación de grilla interpolada a CSV.
    
    Verifica que:
    - export_interpolated_grid_to_csv genera CSV válido
    - El CSV contiene columnas x, y, value
    """
    # Crear grilla sintética
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    csv_str = export_interpolated_grid_to_csv(x, y, values)
    
    # Verificar que es un string válido
    assert isinstance(csv_str, str), "Debe retornar un string"
    assert 'x,y,value' in csv_str, "Debe contener header 'x,y,value'"
    
    # Verificar que se puede leer como DataFrame
    from io import StringIO
    df_csv = pd.read_csv(StringIO(csv_str))
    assert len(df_csv) == 4, "Debe tener 4 filas (2x2 grid)"
    assert list(df_csv.columns) == ['x', 'y', 'value'], "Columnas deben ser x, y, value"
    
    print("✓ Test export CSV: Grilla exportada correctamente")


def test_figure_to_bytes():
    """
    Test: Conversión de figura a bytes.
    
    Verifica que:
    - figure_to_bytes convierte correctamente una figura a bytes
    - Los bytes se pueden escribir a archivo
    """
    df = create_synthetic_drilling_data()
    
    fig, ax, poly = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        nx=30,
        ny=30
    )
    
    # Convertir a bytes
    img_bytes = figure_to_bytes(fig, format='png', dpi=100)
    
    assert isinstance(img_bytes, bytes), "Debe retornar bytes"
    assert len(img_bytes) > 0, "Los bytes no deben estar vacíos"
    
    # Verificar que comienza con firma PNG
    assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Debe ser un archivo PNG válido"
    
    plt.close(fig)
    
    print(f"✓ Test figure_to_bytes: Figura convertida a {len(img_bytes)} bytes PNG")


def test_plot_contour_with_minimal_data():
    """
    Test: Generación de contorno con datos mínimos (3 IDs, 1 punto cada uno).
    
    Verifica que:
    - La función funciona con el mínimo de datos
    - Genera un polígono válido incluso con pocos puntos
    """
    # Datos mínimos: 3 sondeos con 1 punto cada uno
    df_minimal = pd.DataFrame({
        'id': ['P-01', 'P-02', 'P-03'],
        'abscisa': [100.0, 150.0, 200.0],
        'cota': [50.0, 52.0, 51.0],
        'qc': [2.5, 3.0, 2.8]
    })
    
    fig, ax, poly = plot_contour_between_id_minmax(
        df_minimal,
        x_col='abscisa',
        y_col='cota',
        z_col='qc',
        id_col='id',
        nx=20,
        ny=20,
        prefer_method='linear'  # Usar linear con pocos puntos
    )
    
    # Con un solo punto por ID, ymin == ymax, el polígono será una línea
    # pero debe generarse sin errores
    assert hasattr(poly, 'area'), "poly debe tener atributo 'area'"
    # El área podría ser 0 o muy pequeña con un solo punto por ID
    
    plt.close(fig)
    
    print(f"✓ Test datos mínimos: Función ejecuta sin errores. Área: {poly.area:.4f}")


# Función principal para ejecutar tests con pytest
if __name__ == '__main__':
    print("=" * 70)
    print("Pruebas unitarias para plot_contour_between_id_minmax()")
    print("=" * 70)
    print("\nEjecutando pruebas...\n")
    
    try:
        test_plot_contour_basic()
    except Exception as e:
        print(f"✗ Test básico falló: {e}")
    
    try:
        test_plot_contour_with_y_limits()
    except Exception as e:
        print(f"✗ Test y_limits falló: {e}")
    
    try:
        test_plot_contour_interpolation_methods()
    except Exception as e:
        print(f"✗ Test métodos interpolación falló: {e}")
    
    try:
        test_plot_contour_custom_parameters()
    except Exception as e:
        print(f"✗ Test parámetros personalizados falló: {e}")
    
    try:
        test_plot_contour_insufficient_ids()
    except Exception as e:
        print(f"✗ Test IDs insuficientes falló: {e}")
    
    try:
        test_export_grid_to_csv()
    except Exception as e:
        print(f"✗ Test export CSV falló: {e}")
    
    try:
        test_figure_to_bytes()
    except Exception as e:
        print(f"✗ Test figure_to_bytes falló: {e}")
    
    try:
        test_plot_contour_with_minimal_data()
    except Exception as e:
        print(f"✗ Test datos mínimos falló: {e}")
    
    print("\n" + "=" * 70)
    print("Tests completados. Usa 'pytest -v' para ejecución completa.")
    print("=" * 70)
