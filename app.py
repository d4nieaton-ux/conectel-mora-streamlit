import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_mora.pkl")
scaler = joblib.load("scaler.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

st.title("Predicción de Riesgo de Morosidad - ConecTel")

st.write("Ingrese los datos del cliente para estimar su probabilidad de caer en mora superior a 90 días.")

edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
antiguedad_meses = st.number_input("Antigüedad en meses", min_value=0, value=24)

region = st.selectbox("Región", [
    "Antofagasta", "Araucanía", "Atacama", "Biobío", "Coquimbo",
    "Los Lagos", "Maule", "Metropolitana", "O'Higgins", "Valparaíso"
])

genero = st.selectbox("Género", [
    "Femenino", "Masculino", "No binario", "Prefiero no decir"
])

tipo_contrato = st.selectbox("Tipo de contrato", [
    "Anual", "Bianual", "Mensual"
])

plan = st.selectbox("Plan", [
    "Básico", "Estándar", "Premium"
])

metodo_pago = st.selectbox("Método de pago", [
    "Cheque", "Débito automático", "Efectivo", "Transferencia", "WebPay"
])

descuento_activo = st.selectbox("Descuento activo", ["No", "Sí"])

tiene_internet = st.selectbox("Tiene internet", [0, 1])
velocidad_mbps = st.number_input("Velocidad Mbps", min_value=0, value=100)

tiene_tv = st.selectbox("Tiene TV", [0, 1])
tiene_linea_movil = st.selectbox("Tiene línea móvil", [0, 1])

num_servicios = tiene_internet + tiene_tv + tiene_linea_movil

factura_mensual_clp = st.number_input("Factura mensual CLP", min_value=0, value=50000)
ingreso_estimado_clp = st.number_input("Ingreso estimado CLP", min_value=1, value=800000)

dias_mora_hist = st.number_input("Días de mora histórica", min_value=0, value=0)
reclamos_12m = st.number_input("Reclamos últimos 12 meses", min_value=0, value=0)
llamadas_soporte_6m = st.number_input("Llamadas soporte últimos 6 meses", min_value=0, value=1)

nps = st.number_input("NPS", min_value=1, max_value=10, value=7)
meses_sin_reajuste = st.number_input("Meses sin reajuste", min_value=0, value=6)
cambios_plan_12m = st.number_input("Cambios de plan últimos 12 meses", min_value=0, value=0)

if st.button("Predecir riesgo"):

    ratio_factura_ingreso = factura_mensual_clp / ingreso_estimado_clp
    indice_conflictividad = reclamos_12m + llamadas_soporte_6m
    indice_servicios = num_servicios
    ratio_mora_antiguedad = dias_mora_hist / (antiguedad_meses + 1)

    datos = pd.DataFrame([{
        "edad": edad,
        "antiguedad_meses": antiguedad_meses,
        "tiene_internet": tiene_internet,
        "velocidad_mbps": velocidad_mbps,
        "tiene_tv": tiene_tv,
        "tiene_linea_movil": tiene_linea_movil,
        "num_servicios": num_servicios,
        "factura_mensual_clp": factura_mensual_clp,
        "dias_mora_hist": dias_mora_hist,
        "reclamos_12m": reclamos_12m,
        "llamadas_soporte_6m": llamadas_soporte_6m,
        "nps": nps,
        "meses_sin_reajuste": meses_sin_reajuste,
        "ingreso_estimado_clp": ingreso_estimado_clp,
        "cambios_plan_12m": cambios_plan_12m,
        "ratio_factura_ingreso": ratio_factura_ingreso,
        "indice_conflictividad": indice_conflictividad,
        "indice_servicios": indice_servicios,
        "ratio_mora_antiguedad": ratio_mora_antiguedad,
    }])

    datos = pd.get_dummies(datos)

    datos = datos.reindex(
        columns=columnas_modelo,
        fill_value=0
    )

    datos_scaled = scaler.transform(datos)

    probabilidad = modelo.predict_proba(datos_scaled)[0][1]
    porcentaje = probabilidad * 100

    if porcentaje >= 70:
        riesgo = "Alto"
    elif porcentaje >= 40:
        riesgo = "Medio"
    else:
        riesgo = "Bajo"

    st.subheader("Resultado de la predicción")
    st.write(f"Probabilidad estimada de mora: **{porcentaje:.2f}%**")
    st.write(f"Nivel de riesgo: **{riesgo}**")