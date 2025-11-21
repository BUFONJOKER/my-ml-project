import gradio as gr
import pandas as pd
import os
from github import Github
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import numpy as np

# --- 1. Train/Load Model (Simplified for Demo) ---
# In a real app, you would load 'model.pkl' from disk
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# --- 2. Helper Functions ---
def predict_flower(sepal_len, sepal_wid, petal_len, petal_wid):
    prediction = model.predict([[sepal_len, sepal_wid, petal_len, petal_wid]])
    return f"Predicted Species: {data.target_names[prediction[0]]}"

def upload_and_trigger_retrain(file_obj):
    if file_obj is None:
        return "⚠️ No file uploaded."
    
    # Get Token from HF Secrets
    token = os.getenv("GH_TOKEN")
    if not token:
        return "❌ Error: GH_TOKEN secret is missing in Hugging Face settings."

    try:
        # Connect to GitHub
        g = Github(token)
        # UPDATE THIS LINE with your details!
        repo = g.get_repo("BUFONJOKER/my-ml-project") 
        
        # Read the uploaded CSV content
        with open(file_obj.name, "r") as f:
            new_content = f.read()

        # Define path in repo
        file_path = "data/dataset.csv"
        
        # Check if file exists to decide 'create' or 'update'
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(file_path, "Update data via HF Admin Panel", new_content, contents.sha)
            action = "Updated"
        except:
            repo.create_file(file_path, "Create data via HF Admin Panel", new_content)
            action = "Created"
            
        return f"✅ Success! {action} 'data/dataset.csv' on GitHub. \n🚀 Training Pipeline triggered!"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- 3. Build App with Tabs ---
with gr.Blocks(title="Iris ML App") as app:
    gr.Markdown("# 🌸 Iris Flower AI System")
    
    with gr.Tab("🔮 Prediction"):
        with gr.Row():
            sl = gr.Number(label="Sepal Length")
            sw = gr.Number(label="Sepal Width")
            pl = gr.Number(label="Petal Length")
            pw = gr.Number(label="Petal Width")
        btn = gr.Button("Predict", variant="primary")
        out = gr.Textbox(label="Result")
        btn.click(predict_flower, [sl, sw, pl, pw], out)

    with gr.Tab("⚙️ Admin Control"):
        gr.Markdown("### 📂 Upload New Data to Retrain Model")
        gr.Markdown("*Uploading here will push to GitHub and start the training workflow.*")
        
        file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
        upload_btn = gr.Button("Upload & Train", variant="stop")
        status_out = gr.Textbox(label="System Logs")
        
        upload_btn.click(upload_and_trigger_retrain, file_input, status_out)

# Add password protection to the whole app (Optional but recommended)
# app.launch(auth=("admin", "pass123")) 
app.launch()