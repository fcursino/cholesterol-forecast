import gradio as gr
import joblib
import pandas as pd
import os

port = int(os.environ.get('PORT', 10000))
modelo = joblib.load('./modelo_colesterol.pkl')

def predict(grupo_sanguineo, fumante, nivel_atividade_fisica, idade, peso, altura):
  _fumante = 'Sim' if fumante else 'Não'
  predicao_individual = {
    'grupo_sanguineo': grupo_sanguineo,
    'fumante': _fumante,
    'nivel_atividade_fisica': nivel_atividade_fisica,
    'idade': idade,
    'peso': peso,
    'altura': altura
  }
  predict_df = pd.DataFrame(predicao_individual, index=[1])
  colesterol = modelo.predict(predict_df)
  return colesterol.reshape(-1)

demo = gr.Interface(
  fn=predict,
  inputs=[
    gr.Radio(['O','A','B','AB']),
    'checkbox',
    gr.Radio(['Baixo','Moderado','Alto']),
    gr.Slider(20, 80, step=1),
    gr.Slider(40, 160, step=0.1),
    gr.Slider(150, 200, step=1)
  ],
  outputs=['number']
)

demo.launch(server_port=port)