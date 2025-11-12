import joblib
import numpy as np
def save_model(model, path):
    joblib.dump(model, path)
def load_model(path):
    return joblib.load(path)
