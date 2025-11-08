# Interpolación 2D de Parámetros Geotécnicos - Función plot_contour_between_id_minmax

## 🎯 Descripción

Esta aplicación Streamlit genera **mapas de contorno 2D** de parámetros geotécnicos limitados por un **polígono envolvente** construido a partir de las cotas mínimas y máximas de cada sondeo.

### Características Principales

- ✅ **Interpolación por sondeo**: Genera contornos limitados por un polígono basado en min/max de cotas por ID
- ✅ **Mapeo flexible**: Selección interactiva de columnas X, Y, Z e ID
- ✅ **Múltiples parámetros**: Interpola uno o varios parámetros simultáneamente
- ✅ **Configuración completa**: Control total sobre método, resolución, niveles y visualización
- ✅ **Exportación robusta**: Descarga PNG (300 dpi), CSV (grilla interpolada) y GeoJSON (polígono)
- ✅ **Ejemplo incluido**: Botón para cargar datos de ejemplo y probar la aplicación
- ✅ **Soporte shapely**: Usa shapely para cálculos geométricos (con fallback a matplotlib.path)

## 📋 Requisitos

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/camilogardi/Profiles.git
cd Profiles

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r streamlit_app/requirements.txt

# 4. (Opcional pero recomendado) Instalar shapely para mejor rendimiento
pip install shapely>=2.0.0
```

### Dependencias Principales

- Python >= 3.8
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- openpyxl >= 3.0.0 (soporte Excel .xlsx)
- xlrd >= 2.0.1 (soporte Excel .xls)
- shapely >= 2.0.0 (recomendado, mejora máscaras; tiene fallback a matplotlib.path)

## 🚀 Ejecución

### Aplicación principal con plot_contour_between_id_minmax

```bash
streamlit run streamlit_app/app_contour_by_id.py
```

La aplicación se abrirá en `http://localhost:8501`

### Aplicación original (interpolación general)

```bash
streamlit run streamlit_app/app.py
```

## 📁 Formato del Archivo de Entrada

### Requisitos

El archivo debe contener:

1. **Columna X** (abscisa/coordenada Este): Valores numéricos
2. **Columna Y** (cota/elevación): Valores numéricos
3. **Columna ID** (identificador de sondeo): **OBLIGATORIO** para plot_contour_between_id_minmax
4. **Una o más columnas de parámetros**: Valores numéricos a interpolar

### Ejemplo: `streamlit_app/examples/example_table.csv`

```csv
id,abscisa,cota,qc,gamma,LL,IP,humedad
P-01,100.0,50.5,2.5,18.5,35.2,12.5,22.3
P-01,100.0,48.0,3.2,19.1,38.4,14.2,25.1
P-01,100.0,45.5,4.1,19.8,42.1,16.8,28.4
P-02,150.0,52.3,2.8,18.7,36.1,13.1,23.2
...
```

En este ejemplo:
- `id` = Identificador de sondeo (**OBLIGATORIO**)
- `abscisa` = Coordenada X
- `cota` = Coordenada Y/elevación
- `qc`, `gamma`, `LL`, `IP`, `humedad` = Parámetros a interpolar

## 🎨 Flujo de Uso

1. **Cargar datos**: Sube tu CSV/Excel o presiona "Cargar ejemplo"
2. **Mapear columnas**: Indica qué columnas son X, Y, ID
3. **Seleccionar parámetros**: Elige qué parámetro(s) interpolar
4. **Validar datos**: Revisa estadísticas y warnings
5. **Configurar interpolación**: Ajusta método, resolución, visualización (sidebar)
6. **Generar contornos**: Presiona el botón para crear los mapas
7. **Exportar resultados**: Descarga PNG, CSV y GeoJSON

## 🔧 Función plot_contour_between_id_minmax

### Descripción Técnica

La función `plot_contour_between_id_minmax` genera contornos limitados por un polígono construido de la siguiente manera:

1. **Agrupación**: Los datos se agrupan por `id_col` (identificador de sondeo)
2. **Cálculo**: Para cada ID se calcula:
   - Centroide X (promedio de coordenadas X)
   - Cota mínima (min Y)
   - Cota máxima (max Y)
3. **Ordenamiento**: Los IDs se ordenan por centroide X
4. **Construcción del polígono**:
   - Línea superior: une las cotas máximas de izquierda a derecha
   - Línea inferior: une las cotas mínimas de derecha a izquierda
   - Se cierra el polígono
5. **Interpolación**: Se interpola dentro del polígono usando scipy.interpolate.griddata
6. **Máscara**: Se enmascaran los valores fuera del polígono

### Parámetros Configurables

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | DataFrame con datos |
| `x_col` | str | 'x' | Columna con coordenadas X |
| `y_col` | str | 'y' | Columna con coordenadas Y |
| `z_col` | str | 'z' | Columna con parámetro a interpolar |
| `id_col` | str | 'ID' | Columna con identificadores |
| `y_limits` | tuple/None | None | (ymin, ymax) para limitar rango Y |
| `n_levels` | int | 14 | Número de niveles de contorno |
| `nx` | int | 300 | Resolución grilla en X |
| `ny` | int | 300 | Resolución grilla en Y |
| `cmap` | str | 'viridis' | Mapa de colores |
| `clip_to_range` | bool | True | Recortar interpolación al rango de datos |
| `scatter_size` | int | 8 | Tamaño de puntos de datos |
| `title` | str | 'Interpolación 2D' | Título del gráfico |
| `figsize` | tuple | (10, 6) | Tamaño de figura (ancho, alto) |
| `prefer_method` | str | 'cubic' | Método: 'cubic' o 'linear' |

### Retorno

La función retorna una tupla `(fig, ax, poly)`:

- `fig`: matplotlib.figure.Figure
- `ax`: matplotlib.axes.Axes
- `poly`: shapely.geometry.Polygon (o dummy si shapely no disponible)

### Ejemplo de Uso Programático

```python
import pandas as pd
from streamlit_app.utils import plot_contour_between_id_minmax

# Cargar datos
df = pd.read_csv('datos_sondeos.csv')

# Generar contorno
fig, ax, poly = plot_contour_between_id_minmax(
    df,
    x_col='abscisa',
    y_col='cota',
    z_col='qc',
    id_col='id',
    nx=300,
    ny=300,
    n_levels=14,
    prefer_method='cubic',
    cmap='viridis'
)

# Información del polígono
print(f"Área del polígono: {poly.area:.2f}")
print(f"Bounds: {poly.bounds}")

# Guardar figura
fig.savefig('contorno_qc.png', dpi=300, bbox_inches='tight')
```

## 🧪 Pruebas

### Ejecutar todas las pruebas

```bash
# Instalar pytest si no está instalado
pip install pytest

# Ejecutar todas las pruebas
pytest -v streamlit_app/tests/

# Ejecutar solo pruebas de plot_contour_between_id_minmax
pytest -v streamlit_app/tests/test_plot_contour_between_id_minmax.py

# Ejecutar solo pruebas de read_table
pytest -v streamlit_app/tests/test_read_table.py
```

### Cobertura de Pruebas

**test_plot_contour_between_id_minmax.py** (8 tests):
- ✅ Generación básica de contorno
- ✅ Aplicación de límites Y
- ✅ Métodos de interpolación (linear, cubic)
- ✅ Parámetros personalizados
- ✅ Manejo de IDs insuficientes
- ✅ Exportación a CSV
- ✅ Conversión figura a bytes
- ✅ Datos mínimos (edge case)

**test_read_table.py** (7 tests):
- ✅ Lectura CSV desde BytesIO
- ✅ Lectura XLSX con openpyxl
- ✅ Manejo de archivos XLS
- ✅ Fallback sin extensión
- ✅ Funcionalidad seek() múltiples lecturas
- ✅ Archivos inválidos
- ✅ Mensajes de error cuando falta openpyxl

## 📊 Recomendaciones de Uso

### Datos Mínimos

- **Puntos**: Al menos 3 puntos válidos (con X, Y, ID y parámetro)
- **Sondeos**: Al menos 2 IDs únicos (para formar polígono)
- **Recomendado**: 10+ puntos distribuidos en 3+ sondeos

### Resolución de Grilla

| Uso | nx × ny | Tiempo | Memoria |
|-----|---------|--------|---------|
| Preview rápido | 50×50 | < 1s | Baja |
| Uso general | 100×100 | 1-2s | Media |
| Calidad media | 200×200 | 2-5s | Media |
| Alta calidad | 300×300 | 5-10s | Alta |
| Máximo recomendado | 500×500 | 10-30s | Alta |

⚠️ **Advertencia**: Resoluciones > 500×500 (> 250,000 puntos) pueden causar problemas de memoria

### Método de Interpolación

| Situación | Método Recomendado |
|-----------|--------------------|
| Datos bien distribuidos | cubic |
| Pocos datos (< 10 puntos) | linear |
| Datos ruidosos | linear |
| Máxima suavidad | cubic |

## 🔍 Shapely vs Fallback

### Con Shapely (Recomendado)

```bash
pip install shapely>=2.0.0
```

**Ventajas**:
- Cálculo eficiente de máscaras con `shapely.vectorized.contains`
- Operaciones geométricas robustas (buffer, intersection)
- Soporte para MultiPolygon
- Exportación a GeoJSON

### Sin Shapely (Fallback)

Si shapely no está disponible, la función usa automáticamente `matplotlib.path.Path`:

**Limitaciones**:
- Cálculo de máscara más lento
- Sin operaciones geométricas avanzadas
- Sin exportación GeoJSON
- Polígono dummy para compatibilidad

**Conclusión**: Se recomienda instalar shapely para mejor experiencia

## ⚠️ Solución de Problemas

### Error: "No hay sondajes en df['id']"

**Causa**: No se encontraron datos en la columna ID o la columna no existe

**Solución**:
1. Verifica que hayas seleccionado la columna ID correcta
2. Asegúrate que la columna ID no esté vacía

### Error: "Se requieren al menos 2 sondeos"

**Causa**: Solo hay 1 ID único en los datos

**Solución**:
1. Verifica que tengas datos de múltiples sondeos
2. Revisa que la columna ID contenga valores variados

### Error: "openpyxl not found"

**Causa**: Intentas leer un archivo .xlsx sin openpyxl instalado

**Solución**:
```bash
pip install openpyxl>=3.0.0
```

### Warning: "Resolución muy alta"

**Causa**: nx × ny > 1,000,000 puntos

**Solución**:
1. Reduce nx y/o ny en la configuración
2. Usa 300×300 para la mayoría de casos

## 📚 Estructura del Código

```
streamlit_app/
├── app.py                        # Aplicación original (interpolación general)
├── app_contour_by_id.py         # Nueva aplicación (plot_contour_between_id_minmax)
├── utils.py                     # Funciones utilitarias y plot_contour_between_id_minmax
├── io_helpers.py                # Helpers para UI y validación
├── requirements.txt             # Dependencias
├── examples/
│   └── example_table.csv       # Datos de ejemplo (10 sondeos)
└── tests/
    ├── __init__.py
    ├── test_read_table.py      # Pruebas de lectura de archivos
    └── test_plot_contour_between_id_minmax.py  # Pruebas de función principal
```

## 🔗 Funciones Relacionadas

El módulo `utils.py` también incluye:

- `read_table()`: Lectura robusta de CSV/Excel
- `normalize_column_names()`: Normalización de nombres de columnas
- `get_numeric_columns()`: Detección de columnas numéricas
- `export_interpolated_grid_to_csv()`: Exportación de grilla interpolada
- `figure_to_bytes()`: Conversión de figura matplotlib a bytes
- `polygon_to_geojson()`: Exportación de polígono a GeoJSON

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto. Consulta el archivo LICENSE para más detalles.

## 👤 Autor

**Camilo Gardi**
- GitHub: [@camilogardi](https://github.com/camilogardi)

## 📧 Contacto

Para preguntas, sugerencias o reportes de bugs, por favor abre un issue en GitHub.

---

<div align="center">
  <strong>Interpolación 2D de Parámetros Geotécnicos</strong><br>
  Función plot_contour_between_id_minmax<br>
  Desarrollado con ❤️ usando Python y Streamlit
</div>
