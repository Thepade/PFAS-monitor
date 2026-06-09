import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

import config
from anomaly_detection import detect_anomalies
from alert_engine import determine_risk

# Global variables for caching models
model = None
scaler = None
latest_run_dir = ""


def get_most_recent_run_dir() -> str:
    """Find the most recent run directory based on chronological naming."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, config.RUNS_DIR)
    
    if not os.path.exists(runs_dir):
        return ""
        
    dirs = [
        os.path.join(runs_dir, d) 
        for d in os.listdir(runs_dir) 
        if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("run_")
    ]
    
    if not dirs:
        return ""
        
    # run_{YYYYMMDD}_{HHMMSS} sorts chronologically
    return sorted(dirs)[-1]


def inverse_transform_targets(y_scaled: np.ndarray, current_scaler: Any) -> np.ndarray:
    """Inverse transform just the target column for predictions."""
    n_features = current_scaler.n_features_in_
    target_idx = 0
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
        
    y_inv = np.zeros_like(y_scaled)
    for i in range(y_scaled.shape[1]):
        dummy = np.zeros((len(y_scaled), n_features))
        dummy[:, target_idx] = y_scaled[:, i]
        inv = current_scaler.inverse_transform(dummy)
        y_inv[:, i] = inv[:, target_idx]
    return y_inv


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models and dependencies on startup."""
    global model, scaler, latest_run_dir
    latest_run_dir = get_most_recent_run_dir()
    
    if latest_run_dir:
        model_path = os.path.join(latest_run_dir, "rf_model.pkl")
        scaler_path = os.path.join(latest_run_dir, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"[API] Loaded model and scaler from {latest_run_dir}")
        else:
            print("[API] Warning: Model or scaler not found in the most recent run directory.")
    else:
        print("[API] Warning: No run directory found. Models could not be loaded.")
        
    print(f"[API] PFAS Monitor API running at http://{config.API_HOST}:{config.API_PORT}")
    yield
    # Cleanup on shutdown (if necessary)


app = FastAPI(title="PFAS Monitor API", version="2.0", lifespan=lifespan)


# --- Pydantic Request & Response Models ---

class PredictRequest(BaseModel):
    input_window: List[float]
    site: str
    compound: str

class PredictResponse(BaseModel):
    predicted_concentration_ppt: float
    risk_level: str
    forecast_24h: float
    forecast_48h: float
    forecast_72h: float

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    sites_monitored: List[str]
    api_version: str

class AnomalyResponse(BaseModel):
    site: str
    compound: str
    anomalies: List[bool]
    total_anomalies: int
    latest_severity: str

class MetricResponse(BaseModel):
    Model: str
    RMSE: float
    MAE: float
    R2: float


# --- API Endpoints ---

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Returns the monitoring system and loaded model status."""
    return StatusResponse(
        status="online",
        model_loaded=(model is not None),
        model_type="Random Forest",
        sites_monitored=config.SITES,
        api_version="2.0"
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Accepts hourly window sequences and generates a 72-hour forecast."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is currently unloaded or unavailable.")
        
    if len(req.input_window) != config.WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Input window length must be exactly {config.WINDOW_SIZE} hourly readings.")
        
    # Assemble feature data based on how the scaler was fitted
    n_features = scaler.n_features_in_
    data = np.zeros((config.WINDOW_SIZE, n_features))
    data[:, 0] = req.input_window
    
    # If the model was trained with environmental features, we mock normal values for pure simulation predictability
    if n_features > 1:
        data[:, 1] = 25.0  # Simulated Temperature_C
        data[:, 2] = 7.5   # Simulated pH
        data[:, 3] = 2.75  # Simulated Flow_rate_Ls
        
    # Scale dataset
    data_scaled = scaler.transform(data)
    X = data_scaled.flatten().reshape(1, -1)
    
    # Predict multi-horizon targets
    pred_scaled = model.predict(X)
    pred_inv = inverse_transform_targets(pred_scaled, scaler)[0]
    
    # Interpret specific target horizons correctly
    if len(pred_inv) >= 3:
        f24, f48, f72 = pred_inv[0], pred_inv[1], pred_inv[2]
    else:
        f24 = f48 = f72 = pred_inv[0]
        
    # Primary "predicted" is the immediate 24h block in the sequence structure
    predicted_conc = float(f24)
    risk_level = determine_risk(predicted_conc).upper()
    
    return PredictResponse(
        predicted_concentration_ppt=round(predicted_conc, 2),
        risk_level=risk_level,
        forecast_24h=round(float(f24), 2),
        forecast_48h=round(float(f48), 2),
        forecast_72h=round(float(f72), 2)
    )


@app.get("/anomaly", response_model=AnomalyResponse)
def get_anomaly(site: str = Query(...), compound: str = Query(...)):
    """Retrieves standard data logs, evaluates real-time anomalies and returns severity states."""
    if not latest_run_dir:
        raise HTTPException(status_code=503, detail="No run directory loaded.")
        
    data_path = os.path.join(latest_run_dir, "pfas_sensor_data.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset for anomalies not found.")
        
    try:
        from data_processing import load_data, clean_data, calibrate_signal
        df = load_data(data_path)
        df = clean_data(df)
        df = calibrate_signal(df)
        
        df_filtered = df[(df['site'] == site) & (df['compound'] == compound)].copy()
        if df_filtered.empty:
            raise HTTPException(status_code=404, detail="Requested Site or Compound not found.")
            
        # Dynamically execute isolation forest validation
        df_anom = detect_anomalies(df_filtered)
        
        last_24 = df_anom.tail(24)
        anomalies = last_24['anomaly_flag'].tolist()
        total = df_anom['anomaly_flag'].sum()
        latest_sev = last_24.iloc[-1]['severity']
        
        return AnomalyResponse(
            site=site,
            compound=compound,
            anomalies=anomalies,
            total_anomalies=int(total),
            latest_severity=latest_sev
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report", response_class=PlainTextResponse)
def get_report():
    """Serve the plaintext run report from the active session."""
    if not latest_run_dir:
        raise HTTPException(status_code=503, detail="No run directory loaded.")
        
    report_path = os.path.join(latest_run_dir, config.REPORT_FILENAME)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found.")
        
    with open(report_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/metrics", response_model=List[MetricResponse])
def get_metrics():
    """Provide structured validation accuracy performance indicators natively as JSON."""
    if not latest_run_dir:
        raise HTTPException(status_code=503, detail="No run directory loaded.")
        
    metrics_path = os.path.join(latest_run_dir, "metrics_summary.csv")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics summary not found.")
        
    df = pd.read_csv(metrics_path)
    return [MetricResponse(**r) for r in df.to_dict('records')]


@app.get("/")
def get_dashboard():
    """Serve the main web dashboard interface."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return PlainTextResponse("Dashboard not found.", status_code=404)


@app.get("/static/{filename}")
def get_static(filename: str):
    """Serve static assets for the dashboard."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return PlainTextResponse("File not found.", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
