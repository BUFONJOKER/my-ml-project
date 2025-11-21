import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# --- 1. Load or Train Model (Simple version for demo) ---
# In a real app, you would load 'model.pkl'. 
# For this demo, we train it on the fly to ensure it works instantly.
data = load_iris()
X = data.data
y = data.target
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# --- 2. Define the Prediction Function ---
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    class_name = data.target_names[prediction[0]]
    return f"This is likely an {class_name} flower."

# --- 3. Create Gradio Interface ---
iface = gr.Interface(
    fn=predict_flower,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text",
    title="Iris Flower Predictor",
    description="Enter dimensions to predict the flower species."
)

iface.launch()