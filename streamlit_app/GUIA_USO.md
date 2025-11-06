# Guía de Uso - Generador de Perfiles Geotécnicos

## Inicio Rápido

### Ejecutar la Aplicación

```bash
# Navegar al directorio del proyecto
cd Profiles

# Activar entorno virtual (si corresponde)
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Ejecutar la aplicación
streamlit run streamlit_app/app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Flujo de Trabajo Paso a Paso

### Paso 1: Preparar tus Archivos

Necesitas **DOS archivos** en formato CSV o Excel:

#### Archivo A: Cabeceras de Sondeos
Contiene información básica de cada sondeo:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| ID | Identificador único del sondeo | S-01, S-02, etc. |
| x | Coordenada Este (Easting) | 100, 150, 200 |
| y | Coordenada Norte (Northing) | 200, 205, 198 |
| cota | Elevación de superficie (m) | 50.5, 52.3, 51.8 |

**Ejemplo:** `examples/example_headers.csv`

```csv
ID,x,y,cota
S-01,100,200,50.5
S-02,150,205,52.3
S-03,200,198,51.8
```

#### Archivo B: Ensayos por Profundidad
Contiene mediciones a diferentes profundidades:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| ID | ID del sondeo (debe coincidir con Archivo A) | S-01 |
| profundidad | Profundidad desde la cota (m) | 0.5, 2.0, 4.0 |
| parámetro(s) | Valores medidos | SPT, peso_unitario, etc. |

**Ejemplo:** `examples/example_samples.csv`

```csv
ID,profundidad,peso_unitario,limite_liquido,SPT,humedad
S-01,0.5,18.5,35,10,22
S-01,2.0,19.2,38,15,25
S-01,4.0,19.8,42,18,28
```

**Nota importante:** La profundidad debe ser **positiva hacia abajo** desde la cota.

### Paso 2: Cargar Archivos en la Aplicación

1. Abre la aplicación en tu navegador
2. En la sección "Paso 1: Cargar archivos":
   - Haz clic en **"Browse files"** bajo "Archivo A: Cabeceras"
   - Selecciona tu archivo de cabeceras
   - Haz clic en **"Browse files"** bajo "Archivo B: Ensayos"
   - Selecciona tu archivo de ensayos
3. Verás una vista previa de ambos archivos

### Paso 3: Mapear Columnas

1. En la sección "Paso 2: Mapear columnas":
   - **Archivo de cabeceras:**
     - Selecciona la columna que contiene el ID del sondeo
     - Selecciona las columnas de coordenadas X, Y
     - Selecciona la columna de cota (elevación)
   - **Archivo de ensayos:**
     - Selecciona la columna con el ID del sondeo
     - Selecciona la columna de profundidad
2. Presiona **"Aplicar mapeo y procesar datos"**

### Paso 4: Revisar Datos Procesados

Después del mapeo, verás:

- Número de ensayos procesados
- Número de sondeos únicos
- Tabla resumen con límites verticales de cada sondeo:
  - `z_top`: Elevación de superficie
  - `z_bottom`: Elevación del fondo
  - `max_profundidad`: Profundidad máxima alcanzada
  - `n_ensayos`: Cantidad de ensayos en ese sondeo

**Verifica:**
- ✓ Que no haya advertencias sobre IDs sin coincidencia
- ✓ Que las elevaciones tengan sentido (z_top > z_bottom)
- ✓ Que haya suficientes ensayos por sondeo

### Paso 5: Configurar Visualización

Usa el **panel lateral izquierdo** para ajustar:

#### 📊 Parámetro a Graficar
Selecciona qué columna de parámetro quieres visualizar (ej: SPT, peso_unitario, etc.)

#### 📊 Ordenación de Sondeos (Eje X)

Elige cómo ordenar los sondeos horizontalmente:

- **Coordenada X real:** Usa las coordenadas X directamente (útil si sondeos están en línea E-W)
- **Ordenar por X, luego Y:** Ordena primero por X, desempata por Y (útil para sondeos en grilla)
- **Proyección PCA:** Proyecta sobre el eje principal (recomendado para transectos oblicuos)

**¿Cuándo usar PCA?**
- Cuando los sondeos forman una línea diagonal
- Cuando quieres el perfil a lo largo de la dirección de máxima variación
- Cuando los sondeos no están alineados con ejes cardinales

#### 🔲 Resolución de Grilla

- **Puntos en X:** 20-1000 (default: 200)
  - Mayor = más detalle horizontal, más tiempo de cálculo
- **Puntos en Z:** 20-1000 (default: 200)
  - Mayor = más detalle vertical, más tiempo de cálculo

**Recomendación:**
- Para pruebas rápidas: 50-100 puntos
- Para visualización final: 200-300 puntos

#### 🎨 Método de Interpolación

Elige el método de interpolación:

| Método | Características | Cuándo usar |
|--------|-----------------|-------------|
| **griddata_linear** | Rápido, suave, equilibrado | Uso general, primera opción |
| **griddata_nearest** | Preserva valores discretos | Datos categóricos, zonas |
| **griddata_cubic** | Muy suave, puede sobrepasar | Variaciones suaves |
| **rbf** | Flexible, varias funciones | Datos dispersos, suaves |
| **idw** | Simple, robusto, sin sobrepaso | Datos irregulares |

**Parámetros adicionales:**
- **RBF:** Elige función (multiquadric, gaussian, etc.)
- **IDW:** Ajusta potencia (1.0-5.0, default: 2.0)
  - Mayor potencia = más localizado (influencia menor a distancia)

#### 🎨 Visualización

- **Niveles de contorno:** 5-50 (más niveles = más detalle, pero puede ser ruidoso)
- **Mapa de colores:** Elige entre 180+ opciones de matplotlib
  - Recomendados: viridis, plasma, coolwarm, RdYlBu

#### 🔍 Enmascaramiento

- **Aplicar máscara vertical:** ✓ Recomendado
  - Evita interpolar donde no hay datos reales
  - Solo calcula valores en zonas con cobertura de sondeos
- **Distancia horizontal máxima:** 
  - 0 = automático (1.5× distancia al sondeo más cercano)
  - Valor fijo = define radio de influencia manualmente

#### 📝 Anotaciones

- **Mostrar etiquetas de sondeos:** Muestra ID de cada sondeo en el perfil
- **Mostrar puntos de ensayo:** Marca ubicación de cada medición

### Paso 6: Generar Perfil

1. Presiona el botón grande **"Generar perfil"**
2. La aplicación:
   - Calculará las posiciones X de los sondeos
   - Interpolará valores en la grilla X-Z
   - Aplicará la máscara vertical
   - Generará la figura
3. Verás el perfil con:
   - Contornos coloreados del parámetro
   - Líneas verticales mostrando extensión de cada sondeo
   - Etiquetas de sondeos (si activado)
   - Puntos de ensayo (si activado)
   - Eje X: Posición horizontal (según método elegido)
   - Eje Z: Elevación (cota)

**Métricas mostradas:**
- Puntos interpolados: Número de celdas válidas
- Valor mínimo/máximo: Rango del parámetro interpolado

### Paso 7: Exportar Resultados

En la sección "Paso 6: Exportar resultados":

#### Descargar Figura (PNG)
- Alta resolución: 300 DPI
- Formato: PNG con transparencia
- Incluye todos los elementos visuales
- Listo para publicación/informe

#### Descargar Grilla (CSV)
- Formato: X, Z, Value
- Solo incluye celdas válidas (sin NaN)
- Útil para análisis posterior o GIS
- Importable en Excel, QGIS, ArcGIS, etc.

## Ejemplos de Uso

### Ejemplo 1: Perfil de SPT

```
1. Cargar: example_headers.csv + example_samples.csv
2. Mapear: ID, x, y, cota, profundidad
3. Seleccionar parámetro: SPT
4. Método: PCA
5. Interpolación: IDW (power=2.0)
6. Generar y exportar
```

**Resultado:** Perfil vertical mostrando variación de SPT con profundidad y posición.

### Ejemplo 2: Peso Unitario con Máscara Estricta

```
1. Cargar archivos
2. Mapear columnas
3. Seleccionar parámetro: peso_unitario
4. Método ordenación: Coordenada X real
5. Interpolación: griddata_linear
6. Máscara: distancia máxima = 50m
7. Generar
```

**Resultado:** Perfil conservador que solo interpola cerca de sondeos.

## Resolución de Problemas

### Error: "No se encontraron columnas de parámetros"

**Causa:** El archivo B no tiene columnas numéricas además de ID y profundidad.

**Solución:**
- Verifica que el archivo B tenga al menos una columna de parámetro
- Asegúrate de que los valores sean numéricos (no texto)

### Error: "IDs sin coincidencia"

**Causa:** Algunos IDs en archivo B no existen en archivo A.

**Solución:**
- Revisa que los IDs sean exactamente iguales (mayúsculas, espacios)
- Verifica que no haya errores de tipeo
- La aplicación continuará con los IDs que sí coincidan

### Advertencia: "Pocos valores finitos"

**Causa:** La interpolación produjo muchos NaN (máscara muy restrictiva o datos muy dispersos).

**Solución:**
- Desactiva temporalmente la máscara vertical
- Prueba con otro método de interpolación (nearest o IDW)
- Aumenta la distancia horizontal máxima

### Figura aparece "vacía" o con muchos huecos

**Causa:** Método de interpolación no adecuado o configuración de máscara.

**Solución:**
- Prueba con `griddata_nearest` (siempre funciona)
- Reduce el número de niveles de contorno
- Verifica que los datos tengan suficiente cobertura

### Out of Memory / Aplicación lenta

**Causa:** Resolución de grilla muy alta.

**Solución:**
- Reduce nx y nz a 50-100 para pruebas
- Cierra otras aplicaciones
- Para producciones finales, usa máximo 300-500 puntos

### Orden de sondeos no lógico

**Causa:** Método de ordenación no apropiado para tu geometría.

**Solución:**
- Si sondeos en línea E-W: usa "Coordenada X real"
- Si sondeos en diagonal: usa "Proyección PCA"
- Si sondeos en grilla: usa "Ordenar por X, luego Y"

## Consejos y Buenas Prácticas

### Preparación de Datos

✓ **Limpia tus datos antes:** Elimina filas vacías, verifica tipos de datos
✓ **Usa IDs consistentes:** Mismo formato en ambos archivos
✓ **Verifica coordenadas:** Que tengan sentido geográficamente
✓ **Revisa profundidades:** Deben ser positivas hacia abajo

### Configuración Óptima

✓ **Primera exploración:**
- Resolución baja (50x50)
- Método rápido (griddata_linear)
- Ver distribución de datos

✓ **Visualización final:**
- Resolución media-alta (200x200)
- Método apropiado según datos
- Activar todas las anotaciones

### Interpretación de Resultados

✓ **Zonas enmascaradas (blanco/vacío):** No hay datos suficientes para interpolar
✓ **Contornos muy rectos:** Puede indicar extrapolación, verificar máscara
✓ **Valores extremos:** Revisar si son datos reales o artefactos de interpolación

### Documentación

✓ **Guarda configuración:** Anota método y parámetros usados
✓ **Exporta ambos:** Figura (PNG) y datos (CSV)
✓ **Incluye metadatos:** Fecha, proyecto, responsable

## Preguntas Frecuentes

**P: ¿Puedo tener sondeos con diferentes números de ensayos?**
R: Sí, perfectamente. Cada sondeo puede tener distinta cantidad de mediciones.

**P: ¿Los sondeos deben estar alineados?**
R: No necesariamente. La proyección PCA funciona bien con geometrías irregulares.

**P: ¿Puedo usar coordenadas UTM o geográficas?**
R: Sí, cualquier sistema de coordenadas funciona. Solo afecta la escala del eje X.

**P: ¿Qué pasa si tengo huecos en los datos?**
R: La interpolación rellenará los huecos, pero la máscara puede marcarlos como inválidos si están fuera de la cobertura vertical.

**P: ¿Puedo generar múltiples perfiles?**
R: Actualmente un perfil a la vez. Para múltiples parámetros, genera uno, exporta, cambia parámetro, repite.

**P: ¿Soporta 3D?**
R: Esta versión genera perfiles 2D (X-Z). Ver extensiones posibles en README.md para visualización 3D.

## Contacto y Soporte

Para reportar problemas o sugerir mejoras:
- Abre un issue en GitHub
- Incluye descripción del problema
- Adjunta archivos de ejemplo (si es posible)
- Indica configuración usada

---

**Última actualización:** Noviembre 2025
