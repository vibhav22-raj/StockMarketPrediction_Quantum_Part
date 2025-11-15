# quantum_stock_prediction_pennylane.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# Globals / Cache
# -------------------------
# Avoid recreating heavy objects repeatedly on multiple requests
_MODEL_CACHE = {
    "quantum": None,
    "classical": None
}

# ============================================================
# 1️⃣ Load or Create Stock Data (safe & limited size)
# ============================================================
def load_data(file_path: str, n_samples: int = 100):
    """Load CSV (or create synthetic data) and return scaled arrays + scalers."""
    if not os.path.exists(file_path):
        # create small synthetic dataset for demo
        np.random.seed(42)
        data = pd.DataFrame({
            "Open": np.random.uniform(100, 200, n_samples),
            "High": np.random.uniform(100, 210, n_samples),
            "Low": np.random.uniform(90, 195, n_samples),
            "Volume": np.random.uniform(100000, 500000, n_samples),
            "Close": np.random.uniform(100, 200, n_samples)
        })
        data.to_csv(file_path, index=False)
        print(f"[data] Synthetic dataset created at {file_path}")
    else:
        data = pd.read_csv(file_path)
        # normalize column names and map common alternatives
        data.columns = data.columns.str.strip().str.title()
        mapping = {
            "Open Price": "Open",
            "High Price": "High",
            "Low Price": "Low",
            "Close Price": "Close",
            "Vol": "Volume",
            "Adj Close": "Close",
            "Target_Close": "Close"
        }
        data.rename(columns=mapping, inplace=True)

    required_columns = ["Open", "High", "Low", "Volume", "Close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}. Available: {list(data.columns)}")

    # limit rows for speed
    data = data[required_columns].dropna().head(n_samples)
    X = data[["Open", "High", "Low", "Volume"]].values.astype(np.float32)
    y = data[["Close"]].values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y


# ============================================================
# 2️⃣ Quantum Circuit wrapper
# ============================================================
class QuantumCircuit:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # default.qubit is good for local simulation
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, inputs, weights):
        """
        inputs: length n_qubits
        weights: shape (n_layers, n_qubits, 2)
        returns expectation value (scalar)
        """
        # encode inputs into Y rotations
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # entangling ladder
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    def get_circuit(self):
        # create a qnode function bound to this instance
        def qnode(inputs, weights):
            return self.circuit(inputs, weights)

        # return a QNode with torch interface
        return qml.QNode(qnode, self.dev, interface="torch", diff_method="parameter-shift")


# ============================================================
# 3️⃣ Hybrid Quantum-Classical Model (PyTorch)
# ============================================================
class HybridQuantumModel(nn.Module):
    def __init__(self, n_qubits: int, n_features: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # small classical embedding -> quantum inputs
        self.fc1 = nn.Linear(n_features, n_qubits)

        qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        qnode = qc.get_circuit()

        # weight_shapes maps the torchlayer param name to shape
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}

        # TorchLayer: produces a module that maps (inputs) -> output
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # small classical post-processing
        self.fc2 = nn.Linear(1, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # x: [batch, n_features]
        x = torch.tanh(self.fc1(x))            # -> [batch, n_qubits]
        batch_size = x.shape[0]

        # quantum layer expects single-row inputs; apply per-sample
        q_outs = []
        for i in range(batch_size):
            # ensure input is float32 and 1D
            inp = x[i].to(dtype=torch.float32)
            q_out = self.qlayer(inp)  # returns tensor scalar
            q_outs.append(q_out)

        xq = torch.stack(q_outs).view(-1, 1)
        x = torch.relu(self.fc2(xq))
        x = self.fc3(x)
        return x


# ============================================================
# 4️⃣ Fast Classical Model
# ============================================================
class FastClassicalModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================
# 5️⃣ Training helper
# ============================================================
def train_model(model, X_train, y_train, epochs: int = 3, lr: float = 1e-2, batch_size: int = 8, device: str = "cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start = time.time()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(dataloader) if len(dataloader) else total
        print(f"[train] epoch {epoch+1}/{epochs} - loss: {avg:.6f}")
    elapsed = time.time() - start
    print(f"[train] finished in {elapsed:.2f}s")
    return model, elapsed


# ============================================================
# 6️⃣ Evaluation helper
# ============================================================
def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).detach().cpu().numpy()

    y_pred_real = scaler_y.inverse_transform(preds)
    y_test_real = scaler_y.inverse_transform(y_test.detach().cpu().numpy())

    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

    return y_pred_real, y_test_real, rmse, mae, r2


# ============================================================
# 7️⃣ Plotting
# ============================================================
def create_plot(y_test_real, y_pred_real, save_path: str):
    plt.figure(figsize=(12, 5))

    # Time series (subset for visibility)
    plt.subplot(1, 2, 1)
    plot_range = min(50, len(y_test_real))
    plt.plot(y_test_real[:plot_range], label="Actual Price", marker='o', markersize=3)
    plt.plot(y_pred_real[:plot_range], label="Predicted Price", linestyle='--', marker='s', markersize=3)
    plt.title("Stock Price Prediction")
    plt.xlabel("Sample index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)

    # Scatter
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_real, y_pred_real, alpha=0.6, s=30)
    mn, mx = np.min(y_test_real), np.max(y_test_real)
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ============================================================
# 8️⃣ Public API: load(file_path, model_type='quantum')
# ============================================================
def load(file_path: str, model_type: str = "quantum", n_samples: int = 100, device: str = "cpu"):
    """
    Primary function for your app to call.
    - file_path: path to uploaded CSV
    - model_type: "quantum" or "classical"
    Returns: dict with metrics and plot path.
    """
    assert model_type in ("quantum", "classical"), "model_type must be 'quantum' or 'classical'"

    # Load data
    X_scaled, y_scaled, scaler_X, scaler_y = load_data(file_path, n_samples=n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    # to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    n_features = X_train.shape[1]

    results = {}
    # -------------------------
    # TRAIN or GET cached model
    # -------------------------
    if model_type == "classical":
        results["model_name"] = "Classical Neural Network"
        # build or reuse
        model = _MODEL_CACHE["classical"]
        if model is None:
            model = FastClassicalModel(n_features)
            _MODEL_CACHE["classical"] = model
        model, train_time = train_model(model, X_train_t, y_train_t, epochs=8, lr=0.01, batch_size=16, device=device)
        y_pred, y_act, rmse, mae, r2 = evaluate_model(model, X_test_t, y_test_t, scaler_y)
        plot_path = create_plot(y_act, y_pred, "classical_results.png")
    else:
        results["model_name"] = "Hybrid Quantum-Classical Model"
        model = _MODEL_CACHE["quantum"]
        # quantum model is small and expensive to create; cache it
        if model is None:
            n_qubits = n_features
            model = HybridQuantumModel(n_qubits=n_qubits, n_features=n_features, n_layers=1)
            _MODEL_CACHE["quantum"] = model

        # quantum training: fewer epochs, smaller batch to keep demo responsive
        model, train_time = train_model(model, X_train_t, y_train_t, epochs=3, lr=0.01, batch_size=8, device=device)
        y_pred, y_act, rmse, mae, r2 = evaluate_model(model, X_test_t, y_test_t, scaler_y)
        plot_path = create_plot(y_act, y_pred, "quantum_results.png")

    # Fill result dict
    results.update({
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "training_time": float(train_time),
        "plot_path": os.path.abspath(plot_path)
    })

    return results


# ============================================================
# 9️⃣ Standalone test
# ============================================================
if __name__ == "__main__":
    print("Standalone test: running default quantum flow on small synthetic data...")
    out = load("stock_data.csv", model_type="quantum", n_samples=100)
    print("Results:", out)
