# ==================================================
# ⚛️ FastAPI Quantum Stock Prediction (Quantum-Only)
# ==================================================
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from quantum_stock_prediction_pennylane import (
    load_data, 
    HybridQuantumModel, 
    train_model, 
    evaluate_model,
    create_plot
)

# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------
app = FastAPI(title="Quantum Stock Predictor")

# Folder setup
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


# --------------------------------------------------
# 🏠 Home Page
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------
# ⚛️ Quantum Prediction Route (Only Quantum Model)
# --------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded dataset
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and preprocess data
        X_scaled, y_scaled, scaler_X, scaler_y = load_data(file_path, n_samples=100)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
        )

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        # --------------------------------------------------
        # ⚛️ Quantum Model
        # --------------------------------------------------
        print("\n⚛️ Using Quantum-Hybrid Model")
        n_qubits = X_train.shape[1]
        model = HybridQuantumModel(n_qubits, n_qubits, n_layers=1)
        epochs = 3
        batch_size = 8
        plot_name = "quantum_result.png"
        model_type = "Quantum-Hybrid Neural Network"

        # --------------------------------------------------
        # TRAIN + EVALUATE
        # --------------------------------------------------
        trained_model, training_time = train_model(
            model, X_train_t, y_train_t, epochs=epochs, lr=0.01, batch_size=batch_size
        )

        y_pred_real, y_test_real, rmse, mae, r2 = evaluate_model(
            trained_model, X_test_t, y_test_t, scaler_y
        )

        plot_path = os.path.join(STATIC_FOLDER, plot_name)
        create_plot(y_test_real, y_pred_real, plot_path)

        # --------------------------------------------------
        # RETURN RESULT PAGE
        # --------------------------------------------------
        return templates.TemplateResponse("result.html", {
            "request": request,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "training_time": round(training_time, 2),
            "model_type": model_type,
            "plot_path": f"/static/{plot_name}"
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error: {error_details}")
        return HTMLResponse(
            f"""
            <html>
                <head><title>Error</title></head>
                <body style="font-family: Arial; padding: 50px;">
                    <h2>❌ Error during processing</h2>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <pre style="background: #f4f4f4; padding: 20px; border-radius: 5px;">
                    {error_details}
                    </pre>
                    <a href="/" style="display: inline-block; margin-top: 20px; padding: 10px 20px; 
                       background: #3498db; color: white; text-decoration: none; border-radius: 5px;">
                       ⬅️ Back to Upload
                    </a>
                </body>
            </html>
            """, 
            status_code=500
        )


# --------------------------------------------------
# 🚀 Run Locally
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
