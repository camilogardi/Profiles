# Guía Rápida - plot_contour_between_id_minmax

## 🚀 Inicio Rápido (3 pasos)

### 1. Instalar
```bash
git clone https://github.com/camilogardi/Profiles.git
cd Profiles
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r streamlit_app/requirements.txt
pip install shapely  # Recomendado
```

### 2. Ejecutar
```bash
streamlit run streamlit_app/app_contour_by_id.py
```

### 3. Usar
1. Click "Cargar ejemplo" o sube tu CSV/Excel
2. Mapea columnas: X, Y, ID
3. Selecciona parámetro(s)
4. Ajusta configuración en sidebar
5. Click "Generar mapas de contorno"
6. Descarga PNG, CSV o GeoJSON

## 📁 Formato de Archivo Requerido

Tu archivo debe tener:
- ✅ Columna **X** (abscisa/Este)
- ✅ Columna **Y** (cota/elevación)  
- ✅ Columna **ID** (sondeo) - **OBLIGATORIO**
- ✅ Una o más columnas con parámetros numéricos

### Ejemplo
```csv
id,abscisa,cota,qc,gamma
P-01,100.0,50.5,2.5,18.5
P-01,100.0,48.0,3.2,19.1
P-02,150.0,52.3,2.8,18.7
P-02,150.0,50.0,3.5,19.3
```

## ⚙️ Configuración Recomendada

| Parámetro | Recomendado | Uso |
|-----------|-------------|-----|
| nx × ny | 300×300 | Uso general |
| Método | cubic | Datos bien distribuidos |
| Niveles | 14 | Visualización clara |
| Cmap | viridis | Científico estándar |

## 🔍 Resolución de Problemas Comunes

### Error: "No hay sondajes"
**Solución:** Selecciona la columna ID correcta

### Error: "Se requieren al menos 2 sondeos"
**Solución:** Tu archivo debe tener mínimo 2 IDs únicos

### Error: "openpyxl not found"
**Solución:** `pip install openpyxl` o exporta a CSV

### Warning: "Resolución muy alta"
**Solución:** Reduce nx o ny a ≤ 500

## 📚 Recursos

- **Documentación completa:** [README_CONTOUR_BY_ID.md](README_CONTOUR_BY_ID.md)
- **Resumen de cambios:** [RESUMEN_CAMBIOS.md](RESUMEN_CAMBIOS.md)
- **Script demo:** `python demo_plot_contour.py`
- **Tests:** `pytest -v streamlit_app/tests/`

## 💡 Consejos Útiles

1. **Usa el ejemplo** para familiarizarte con la app
2. **Empieza con resolución baja** (100×100) para preview rápido
3. **Aumenta resolución** (300×300) para figuras finales
4. **Instala shapely** para mejor rendimiento
5. **Invierte eje Y** si trabajas con profundidad

## 🆘 Soporte

- **Issues:** https://github.com/camilogardi/Profiles/issues
- **Autor:** [@camilogardi](https://github.com/camilogardi)

---

**¡Listo! Ya puedes generar tus mapas de contorno por sondeo** 🎉
