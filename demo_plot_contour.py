#!/usr/bin/env python
"""
Script de demostración de plot_contour_between_id_minmax.

Este script genera un ejemplo de contorno usando la función
plot_contour_between_id_minmax con datos del archivo de ejemplo.

Uso:
    python demo_plot_contour.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streamlit_app'))

import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_contour_between_id_minmax

def main():
    print("=" * 70)
    print("Demostración de plot_contour_between_id_minmax")
    print("=" * 70)
    
    # Cargar datos de ejemplo
    example_path = "streamlit_app/examples/example_table.csv"
    print(f"\n1. Cargando datos de ejemplo: {example_path}")
    df = pd.read_csv(example_path)
    print(f"   ✓ Cargados: {len(df)} filas, {len(df.columns)} columnas")
    print(f"   Columnas: {list(df.columns)}")
    
    # Información de sondeos
    n_ids = df['id'].nunique()
    print(f"\n2. Información de datos:")
    print(f"   - Sondeos únicos: {n_ids}")
    print(f"   - Puntos totales: {len(df)}")
    print(f"   - Rango X (abscisa): {df['abscisa'].min():.1f} - {df['abscisa'].max():.1f}")
    print(f"   - Rango Y (cota): {df['cota'].min():.1f} - {df['cota'].max():.1f}")
    
    # Generar contorno para parámetro 'qc'
    param = 'qc'
    print(f"\n3. Generando contorno para parámetro: '{param}'")
    print(f"   Parámetros:")
    print(f"   - nx=300, ny=300")
    print(f"   - n_levels=14")
    print(f"   - prefer_method='cubic'")
    print(f"   - cmap='viridis'")
    
    fig, ax, poly = plot_contour_between_id_minmax(
        df,
        x_col='abscisa',
        y_col='cota',
        z_col=param,
        id_col='id',
        nx=300,
        ny=300,
        n_levels=14,
        prefer_method='cubic',
        cmap='viridis',
        title=f'Demostración: Contorno de {param} por sondeos'
    )
    
    print(f"   ✓ Contorno generado exitosamente")
    
    # Información del polígono
    print(f"\n4. Información del polígono:")
    print(f"   - Tipo: {poly.geom_type}")
    print(f"   - Área: {poly.area:.2f}")
    if hasattr(poly, 'bounds'):
        bounds = poly.bounds
        print(f"   - Bounds: [{bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f}]")
    
    # Guardar figura
    output_file = "demo_contour_output.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n5. Figura guardada en: {output_file}")
    
    print("\n" + "=" * 70)
    print("✓ Demostración completada exitosamente")
    print("=" * 70)
    
    # Cerrar figura
    plt.close(fig)

if __name__ == '__main__':
    main()
