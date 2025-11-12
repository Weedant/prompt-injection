# ============================================================================
# scripts/run_all.py
# ============================================================================
import subprocess
import sys

print("1. Building embeddings...")
subprocess.run([sys.executable, "-m", "src.features.embedder", "--build", "--batch", "64"])

print("2. Training model...")
subprocess.run([sys.executable, "src/models/train.py"])

print("3. Launching demo...")
subprocess.run(["streamlit", "run", "streamlit_debug.py"])
