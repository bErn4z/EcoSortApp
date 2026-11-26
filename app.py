# -*- coding: utf-8 -*-
"""EcoSort · Meta III — aplicação web de classificação de resíduos.

Carrega uma imagem (ficheiro ou câmara), mostra-a, classifica-a com o modelo de transfer
learning guardado em model/ (formato TFLite) e apresenta a classe prevista e as
probabilidades das 12 classes. Permite testar várias imagens de seguida.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter

MODEL_DIR = Path(__file__).parent / "model"

# Nome em português e ecoponto sugerido para cada classe do dataset
CLASS_INFO = {
    "battery":     ("Pilhas e baterias", "Pilhão"),
    "biological":  ("Resíduos orgânicos", "Contentor de orgânicos (castanho)"),
    "brown-glass": ("Vidro castanho", "Vidrão (verde)"),
    "cardboard":   ("Cartão", "Ecoponto azul"),
    "clothes":     ("Roupa", "Contentor de têxteis"),
    "green-glass": ("Vidro verde", "Vidrão (verde)"),
    "metal":       ("Metal", "Ecoponto amarelo"),
    "paper":       ("Papel", "Ecoponto azul"),
    "plastic":     ("Plástico", "Ecoponto amarelo"),
    "shoes":       ("Calçado", "Contentor de têxteis / calçado"),
    "trash":       ("Lixo indiferenciado", "Contentor de indiferenciados"),
    "white-glass": ("Vidro branco", "Vidrão (verde)"),
}
LOW_CONFIDENCE = 0.50


@st.cache_resource
def load_model():
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    interpreter = Interpreter(model_path=str(MODEL_DIR / "ecosort.tflite"))
    interpreter.allocate_tensors()
    return interpreter, info


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Igual ao treino: RGB, redimensionamento direto para size×size, valores 0–255."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, ...]


def predict(interpreter, x: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]


def show_result(name: str, image: Image.Image, probs: np.ndarray, classes: list[str]):
    order = np.argsort(probs)[::-1]
    best = classes[order[0]]
    label, bin_ = CLASS_INFO.get(best, (best, "—"))
    col_img, col_res = st.columns([1, 1.3])
    with col_img:
        st.image(image, caption=name, use_container_width=True)
    with col_res:
        st.subheader(f"{label}  ·  {probs[order[0]]:.1%}")
        st.write(f"Classe do modelo: `{best}`  \nDestino sugerido: **{bin_}**")
        if probs[order[0]] < LOW_CONFIDENCE:
            st.warning("Confiança baixa: o modelo não tem a certeza. Experimente outra fotografia, "
                       "com o objeto centrado e fundo simples.")
        table = pd.DataFrame({
            "Classe": [f"{CLASS_INFO.get(classes[i], (classes[i],))[0]} ({classes[i]})" for i in order],
            "Probabilidade": probs[order],
        })
        st.bar_chart(table.set_index("Classe"), horizontal=True, height=330)
    return best, float(probs[order[0]])


st.set_page_config(page_title="EcoSort", page_icon="♻️", layout="wide")
st.title("♻️ EcoSort — classificação de resíduos")
st.caption("Inteligência Computacional 2026/27 · Meta III · Bernardo Vasconcelos e Miguel Bernardes")

interpreter, info = load_model()
classes = info["class_names"]
if "PROVISÓRIO" in info.get("modelo", ""):
    st.error("Modelo provisório, usado apenas para testar o deploy: as previsões ainda não têm significado.")

st.write("Carregue uma ou mais fotografias de um resíduo, ou tire uma fotografia com a câmara. "
         "O modelo indica a categoria mais provável e a probabilidade de cada uma das 12 classes.")

tab_file, tab_cam = st.tabs(["📁 Carregar imagens", "📷 Câmara"])
images = []
with tab_file:
    files = st.file_uploader("Imagens (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
    for f in files or []:
        images.append((f.name, Image.open(io.BytesIO(f.getvalue()))))
with tab_cam:
    shot = st.camera_input("Tirar fotografia")
    if shot is not None:
        images.append(("Fotografia da câmara", Image.open(io.BytesIO(shot.getvalue()))))

if "history" not in st.session_state:
    st.session_state.history = []

for name, image in images:
    st.divider()
    probs = predict(interpreter, preprocess(image, info["img_size"]))
    best, conf = show_result(name, ImageOps.exif_transpose(image), probs, classes)
    entry = (name, best, round(conf, 3))
    if entry not in st.session_state.history:
        st.session_state.history.append(entry)

if st.session_state.history:
    with st.sidebar:
        st.header("Histórico desta sessão")
        st.dataframe(pd.DataFrame(st.session_state.history, columns=["Imagem", "Classe", "Confiança"]),
                     hide_index=True, use_container_width=True)
        if st.button("Limpar histórico"):
            st.session_state.history = []
            st.rerun()

with st.expander("Sobre o modelo"):
    st.write(info.get("descricao", info.get("modelo", "")))
    st.write(f"Entrada: {info['img_size']}×{info['img_size']} RGB · {len(classes)} classes: " + ", ".join(classes))
