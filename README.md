# Interpolación 2D de Parámetros Geotécnicos

Una aplicación Streamlit para generar **mapas de contorno 2D** (interpolación espacial) de parámetros geotécnicos a partir de datos de sondeos o mediciones puntuales.

## 🎯 Características Principales

- **Entrada de una sola tabla:**
  - Archivo único CSV o Excel con coordenadas (X, Y) y múltiples columnas de parámetros medidos

- **Mapeo flexible de columnas:**
  - Selección interactiva de columnas para X (abscisa) e Y (cota/elevación)
  - Detección automática de columnas de coordenadas

- **Selección múltiple de parámetros:**
  - Interpola uno o varios parámetros simultáneamente
  - Genera un mapa de contorno independiente por cada parámetro

- **Múltiples métodos de interpolación:**
  - **Griddata**: linear, nearest, cubic (scipy.interpolate.griddata)
  - **RBF** (Radial Basis Function): multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
  - **IDW** (Inverse Distance Weighting): implementación vectorizada con potencia configurable

- **Enmascaramiento para evitar extrapolación:**
  - **ConvexHull**: Enmascara celdas fuera de la envolvente convexa de los datos
  - **Por distancia**: Enmascara celdas lejanas a los puntos de datos (basado en k-vecinos)
  - **Combinado**: Aplica ambas máscaras simultáneamente

- **Configuración avanzada:**
  - Resolución de grilla ajustable (nx, ny)
  - Múltiples paletas de colores
  - Número de niveles de contorno personalizable
  - Opción para invertir eje Y (útil para profundidad)

- **Exportación:**
  - Figuras PNG de alta resolución (300 dpi)
  - Grilla interpolada en formato CSV (X, Y, Value)

## 📋 Requisitos

### Dependencias

```bash
pip install -r streamlit_app/requirements.txt
```

Dependencias principales:
- Python >= 3.8
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- openpyxl >= 3.1.0 (para soporte de Excel)

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/camilogardi/Profiles.git
cd Profiles
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r streamlit_app/requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run streamlit_app/app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📁 Formato del Archivo de Entrada

### Estructura requerida

El archivo debe ser CSV o Excel con las siguientes columnas:

1. **Columna X** (abscisa, coordenada Este): Valores numéricos
2. **Columna Y** (cota, elevación, coordenada Norte): Valores numéricos
3. **Una o más columnas de parámetros**: Valores numéricos a interpolar
4. **Columna ID** (opcional): Identificador de puntos/sondeos

### Ejemplo: `examples/example_table.csv`

```csv
id,abscisa,cota,qc,gamma,LL,IP,humedad
P-01,100.0,50.5,2.5,18.5,35.2,12.5,22.3
P-01,100.0,48.0,3.2,19.1,38.4,14.2,25.1
P-02,150.0,52.3,2.8,18.7,36.1,13.1,23.2
P-02,150.0,50.0,3.5,19.3,39.2,15.0,26.0
P-03,200.0,51.8,2.6,18.6,34.8,12.0,21.8
...
```

En este ejemplo:
- `abscisa` = coordenada X
- `cota` = coordenada Y (elevación)
- `qc`, `gamma`, `LL`, `IP`, `humedad` = parámetros a interpolar
- `id` = identificador (opcional, para etiquetas)

### Nombres de columnas

Los nombres de columnas son flexibles. La aplicación:
- Normaliza automáticamente (elimina espacios, convierte a minúsculas)
- Intenta detectar automáticamente columnas X e Y por palabras clave
- Permite mapeo manual en la interfaz

Palabras clave reconocidas:
- **X**: abscisa, x, este, easting
- **Y**: cota, y, elevacion, elevation, norte, northing

## 🎨 Flujo de Uso

1. **Cargar archivo**: Sube tu CSV o Excel con datos geotécnicos
2. **Mapear columnas**: Indica qué columnas son X e Y
3. **Seleccionar parámetros**: Elige uno o más parámetros a interpolar
4. **Revisar estadísticas**: Verifica la distribución y calidad de los datos
5. **Configurar interpolación**: Ajusta método, resolución, máscaras en el panel lateral
6. **Generar mapas**: Presiona el botón para crear los contornos
7. **Exportar resultados**: Descarga figuras PNG y/o grillas CSV

## ⚙️ Métodos de Interpolación

### Griddata (scipy.interpolate.griddata)

Interpolación basada en triangulación:
- **linear**: Rápida, suave, sin sobrepaso
- **nearest**: Preserva valores discretos, crea regiones escalonadas
- **cubic**: Muy suave, puede producir sobrepaso

**Recomendado para**: Datos bien distribuidos, sin muchos outliers

### RBF (Radial Basis Function)

Interpolación usando funciones de base radial:
- **multiquadric**: Suave, buena para datos dispersos
- **inverse**: Similar a IDW
- **gaussian**: Muy suave, puede subestimar extremos
- **linear**, **cubic**, **quintic**: Diferentes grados de suavidad
- **thin_plate**: Minimiza curvatura

**Recomendado para**: Datos irregularmente espaciados, pocos puntos

### IDW (Inverse Distance Weighting)

Promedio ponderado por distancia inversa:
- Potencia típica: 2.0
- Mayor potencia → más peso a puntos cercanos
- No produce sobrepaso (interpolación exacta en puntos conocidos)

**Recomendado para**: Datos con tendencias locales fuertes

## 🔍 Enmascaramiento

### ¿Por qué enmascarar?

La interpolación puede producir valores no confiables fuera del dominio de los datos reales. El enmascaramiento marca estas zonas como NaN para evitar interpretaciones erróneas.

### Métodos disponibles

1. **ConvexHull** (Envolvente convexa):
   - Enmascara todo fuera del polígono convexo que encierra los datos
   - Requiere al menos 4 puntos no colineales
   - **Ventaja**: Simple, elimina extrapolación obvia
   - **Limitación**: Puede incluir zonas sin datos si el hull es cóncavo

2. **Por distancia**:
   - Enmascara celdas lejanas al vecino más cercano
   - Distancia umbral configurable o automática (basada en distribución de puntos)
   - **Ventaja**: Respeta huecos dentro de los datos
   - **Limitación**: Requiere ajuste de parámetros

3. **Combinado** (recomendado):
   - Aplica ambas máscaras (intersección)
   - Más conservador, mayor confiabilidad
   - **Ventaja**: Combina fortalezas de ambos métodos

### Parámetros de máscara

- **Distancia máxima**: 0 = automático (1.5× percentil 90 de distancias entre puntos)
- Valores típicos: 10-50 (en unidades de tus coordenadas)

## 📊 Recomendaciones

### Datos mínimos

- Al menos **3 puntos válidos** (con X, Y y parámetro definido)
- Para ConvexHull: al menos **4 puntos no colineales**
- Recomendado: **10+ puntos** para interpolación confiable

### Resolución de grilla

- **Baja** (50×50): Previsualización rápida
- **Media** (100×100): Uso general
- **Alta** (200×200+): Figuras finales, detalles finos
- **Límite**: 500×500 (evitar > 1,000,000 puntos de grilla)

### Elección de método

| Situación | Método recomendado |
|-----------|-------------------|
| Datos bien distribuidos, sin huecos | Griddata linear |
| Datos dispersos, pocos puntos | RBF multiquadric |
| Preservar valores discretos | Griddata nearest |
| Suavidad máxima | Griddata cubic o RBF gaussian |
| Tendencias locales fuertes | IDW con power=2-3 |

## 🔧 Extensiones Futuras

El código está modularizado para facilitar extensiones:

### Añadir Kriging (geoestadística)

```python
# En utils.py, añadir:
from pykrige.ok import OrdinaryKriging

def interpolate_kriging(points_xy, values, grid_x, grid_y, variogram_model='linear'):
    ok = OrdinaryKriging(
        points_xy[:, 0], points_xy[:, 1], values,
        variogram_model=variogram_model
    )
    z, ss = ok.execute('grid', grid_x[0, :], grid_y[:, 0])
    return z
```

### Exportar a GeoTIFF

```python
# Requiere: rasterio, affine
import rasterio
from affine import Affine

def export_geotiff(grid_x, grid_y, grid_values, filename, crs='EPSG:4326'):
    # Calcular transformación afín
    transform = Affine.translation(grid_x[0, 0], grid_y[0, 0]) * \
                Affine.scale((grid_x[0, -1] - grid_x[0, 0]) / grid_x.shape[1],
                            (grid_y[-1, 0] - grid_y[0, 0]) / grid_y.shape[0])
    
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=grid_values.shape[0],
        width=grid_values.shape[1],
        count=1,
        dtype=grid_values.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(grid_values, 1)
```

### Visualización interactiva con Plotly

```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Contour(
    x=grid_x[0, :],
    y=grid_y[:, 0],
    z=grid_values,
    colorscale='Viridis'
))
st.plotly_chart(fig)
```

## 📚 Documentación del Código

El código está completamente documentado con docstrings en español siguiendo el formato NumPy:

- **streamlit_app/app.py**: Aplicación principal Streamlit
- **streamlit_app/utils.py**: Funciones de interpolación, máscaras, grillas
- **streamlit_app/io_helpers.py**: Helpers de UI y validación
- **streamlit_app/requirements.txt**: Dependencias del proyecto

Cada función incluye:
- Descripción de propósito
- Parámetros con tipos y descripciones
- Valores de retorno
- Notas sobre limitaciones o casos especiales

## 🧪 Ejemplo de Uso

### Generar contorno de resistencia por punta (qc)

1. Ejecuta la aplicación: `streamlit run streamlit_app/app.py`
2. Sube el archivo `examples/example_table.csv`
3. Mapea columnas:
   - X → `abscisa`
   - Y → `cota`
4. Selecciona parámetro: `qc`
5. Configura en sidebar:
   - Método: Griddata - Linear
   - Resolución: 100×100
   - Máscara: ConvexHull
6. Presiona "🚀 Generar mapas de contorno"
7. Descarga PNG y/o CSV

## ⚠️ Consideraciones

- **Valores faltantes**: Las filas con X o Y faltantes se eliminan automáticamente
- **Columnas no numéricas**: Solo se interpolan columnas con valores numéricos
- **Memoria**: Resoluciones > 300×300 pueden consumir mucha RAM
- **Tiempo de cómputo**: RBF es más lento que griddata o IDW
- **ConvexHull**: Puede fallar con puntos colineales (< 4 puntos únicos)

## 🐛 Solución de Problemas

### Error: "Insufficient points"
- Verifica que tu archivo tenga al menos 3 filas con valores válidos
- Revisa que las columnas X, Y y parámetros sean numéricas

### Error en ConvexHull
- Usa máscara "Por distancia" en lugar de ConvexHull
- Asegúrate de tener al menos 4 puntos no colineales

### Interpolación produce muchos NaN
- Reduce la resolución de grilla
- Ajusta la máscara por distancia (aumenta distancia máxima)
- Verifica distribución espacial de tus datos

### Figuras no se ven bien
- Aumenta resolución de grilla
- Cambia método de interpolación (prueba RBF)
- Ajusta niveles de contorno

## 📄 Licencia

Este proyecto es de código abierto. Consulta el archivo LICENSE para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 👤 Autor

**Camilo Gardi**
- GitHub: [@camilogardi](https://github.com/camilogardi)

## 📧 Contacto

Para preguntas, sugerencias o reportes de bugs, por favor abre un issue en GitHub.

---

<div align="center">
  <strong>Interpolación 2D de Parámetros Geotécnicos</strong><br>
  Desarrollado con ❤️ usando Python y Streamlit
</div>
