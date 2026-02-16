from fastapi import APIRouter
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from model_storage import original_model, X_test, y_test
from core.logger import log_event

router = APIRouter()

def create_model():
    model = Sequential([
        Dense(20, activation='relu', input_shape=(5,)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@router.post("/attack", tags=["Attack"])
def model_stealing_attack():
    stolen_X = X_test[:200]
    stolen_y = (original_model.predict(stolen_X) > 0.5).astype(int).flatten()

    stolen_model = create_model()
    stolen_model.fit(stolen_X, stolen_y, epochs=20, verbose=0)

    stolen_accuracy = accuracy_score(y_test, (stolen_model.predict(X_test) > 0.5).astype(int).flatten())
    log_event("Executed model stealing attack.")

    return {
        "message": "API key validated and model attack simulated",
        "stolen_model_accuracy": round(stolen_accuracy, 4)
    }
