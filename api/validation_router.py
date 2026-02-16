from fastapi import APIRouter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from models.model_storage import original_model, X_test, y_test
from utils.logger import log_event

router = APIRouter()

@router.get("/analyze_logs", tags=["Validation"])
def analyze_logs():
    normal_queries = np.random.rand(600, 5)
    attack_queries = X_test[:200]
    logs = np.vstack((normal_queries, attack_queries))
    labels = np.array(["Normal"] * 600 + ["Suspicious"] * 200)
    df = pd.DataFrame(logs, columns=[f"Feature{i+1}" for i in range(5)])
    df['Label'] = labels
    df = df.sample(frac=1).reset_index(drop=True)
    log_event("Analyzed logs.")
    return df.to_dict(orient="records")

@router.get("/compare", tags=["Validation"])
def compare_models():
    original_accuracy = accuracy_score(y_test, (original_model.predict(X_test) > 0.5).astype(int).flatten())
    simulated_stolen_accuracy = round(np.random.uniform(0.75, 0.85), 4)
    log_event("Compared model performance.")
    return {
        "original_model_accuracy": round(original_accuracy, 4),
        "stolen_model_accuracy": simulated_stolen_accuracy
    }

@router.get("/defenses", tags=["Defenses"])
def defenses():
    log_event("Displayed defense strategies.")
    return {
        "Rate Limiting": "https://colab.research.google.com/drive/1JCxy6mbEc-5XgHbhQdMzySLWqmtVwifP?usp=sharing",
        "Differential Privacy": "https://colab.research.google.com/drive/1_aUKotwACl63gocZiv7EbubZSIn8n2fp?usp=sharing",
        "Model Watermarking": "https://colab.research.google.com/drive/13IbqwZINdPXvUVrr-y6GB1AQcRp9ZBVG?usp=sharing"
    }
