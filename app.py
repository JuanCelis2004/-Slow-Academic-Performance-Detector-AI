import streamlit as st
import joblib
import pandas as pd
import numpy as np

# CARGAR EL CEREBRO DE LA IA
# Asegúrate de que la ruta sea correcta según tu carpeta
model = joblib.load('models/modelo_riesgo.pkl')
scaler = joblib.load('models/scaler_riesgo.pkl')

st.set_page_config(page_title="Asesor Académico IA", page_icon="🎓")

st.title("🎓 Asesor Académico Inteligente")
st.markdown("""
Ingresa los datos actuales del estudiante para obtener un diagnóstico de riesgo basado en IA.
""")

# ENTRADA DE DATOS CON VALORES MÍNIMOS (Placeholders)
col1, col2 = st.columns(2)

with col1:
    p = st.number_input("Promedio actual", min_value=1.0, max_value=5.0, value=1.0, step=0.1, help="Ingresa el promedio del semestre actual")
    a = st.slider("Asistencia (%)", 0, 100, 0)
    t = st.number_input("Tareas entregadas (0-10)", min_value=0, max_value=10, value=0)

with col2:
    np_ = st.number_input("Última Nota Parcial", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    m = st.number_input("Materias inscritas", min_value=1, max_value=10, value=1)
    h = st.number_input("Promedio histórico", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

st.divider()

# Solo se ejecuta el análisis si se presiona el botón
if st.button("🔍 Analizar Situación Académica", use_container_width=True):
    # Ingeniería de Datos interna
    ia = p * (a / 100)
    vh = p - h

    # Crear DataFrame
    datos = pd.DataFrame([[p, a, t, np_, m, h, ia, vh]], 
                         columns=['promedio', 'asistencia', 'tareas', 'nota_parcial', 
                                  'materias', 'historial', 'interaccion_acad', 'variacion_historial'])
    
    # Predicción
    datos_scaled = scaler.transform(datos)
    pred = model.predict(datos_scaled)[0]
    prob = model.predict_proba(datos_scaled)[0][pred]

    # Configuración de alertas
    niveles = {
        0: ("SIN RIESGO ✅", "green", "El estudiante mantiene un ritmo óptimo."),
        1: ("RIESGO MEDIO ⚠️", "orange", "Se recomienda reforzar hábitos de estudio."),
        2: ("RIESGO ALTO 🚨", "red", "Requiere intervención inmediata de bienestar.")
    }
    
    nombre_riesgo, color, mensaje_base = niveles[pred]

    # Mostrar Resultados
    st.subheader(f"Resultado: :{color}[{nombre_riesgo}]")
    st.metric(label="Confianza del Modelo", value=f"{prob*100:.1f}%")
    st.progress(prob)
    
    st.info(mensaje_base)

    # Plan de acción dinámico
    with st.expander("Ver recomendaciones detalladas"):
        if p < 3.0:
            st.write(f"❌ **Académico:** El promedio de {p} está bajo el límite. Priorizar recuperación de notas.")
        if a < 70:
            st.write(f"❌ **Asistencia:** Con {a}%, hay riesgo de pérdida administrativa.")
        if pred == 0:
            st.write("✨ **Excelencia:** Sigue así para mantener tu promedio histórico.")
else:
    st.info("Configura los datos arriba y presiona el botón para ver el diagnóstico.")