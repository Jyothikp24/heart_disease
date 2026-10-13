import joblib
import sys
from sklearn._loss import _loss as sklearn_loss

sys.modules.setdefault("_loss", sklearn_loss)

model_data = joblib.load("heart.joblib")

print("Data type:", type(model_data))

if isinstance(model_data, dict):
    print("Available keys:", model_data.keys())
else:
    print(
        "Feature names:",
        getattr(model_data, "feature_names_in_", None)
    )