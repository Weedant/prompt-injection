# ============================================================================
# scripts/run_embed.py
# ============================================================================
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "src.features.embedder", "--build", "--batch", "64"])