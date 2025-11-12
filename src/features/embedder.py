import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model.encode(texts, show_progress_bar=True)
