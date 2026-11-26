# EcoSort — classificação de resíduos (Meta III)

Aplicação web do projeto EcoSort (Inteligência Computacional 2026/27, ISEC).
Classifica fotografias de resíduos em 12 classes com um modelo de *transfer learning*
e sugere o ecoponto adequado.

**Aplicação pública:** https://ecosortapp-gn7lwjwblo3vxg8wm9gkum.streamlit.app/

Autores: Bernardo Vasconcelos (2024128419) e Miguel Bernardes (2024134199).

## Funcionalidades
- Carregar uma ou várias imagens (ficheiro) ou tirar uma fotografia com a câmara;
- Mostrar a imagem submetida;
- Classificar com o modelo treinado e guardado em `model/`;
- Mostrar a classe prevista, o destino sugerido e as probabilidades das 12 classes;
- Testar várias imagens seguidas (histórico na barra lateral).

## Estrutura
```
app.py                  aplicação Streamlit
requirements.txt        dependências (Streamlit + LiteRT, sem TensorFlow)
model/ecosort.tflite    modelo final exportado para TFLite
model/model_info.json   classes, tamanho de entrada e descrição do modelo
```

## Correr localmente
```
pip install -r requirements.txt
streamlit run app.py
```
