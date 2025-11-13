# ============================================================================
# src/inference/prefilter_service.py
# ============================================================================
import joblib
import re
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

# FIXED: Use consistent filename
MODEL_PATH = ROOT / "models" / "filtered_clf.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.txt"
MODEL_META_PATH = ROOT / "models" / "model_metadata.json"
EMBEDDER_META_PATH = ROOT / "data" / "processed" / "embedder_meta.json"

for p in [MODEL_PATH, THRESHOLD_PATH, MODEL_META_PATH, EMBEDDER_META_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

clf = joblib.load(MODEL_PATH)
threshold = float(THRESHOLD_PATH.read_text().strip())
model_meta = json.loads(MODEL_META_PATH.read_text())
embedder_meta = json.loads(EMBEDDER_META_PATH.read_text())

sbert = SentenceTransformer(embedder_meta["model"])

RULES = [
    # Instruction override
    (r'\bignore\s+(previous|earlier|prior|all)\s+(instructions?|prompts?|commands?)\b', "ignore_instruction"),
    (r'\bdo\s+not\s+(follow|obey|listen\s+to)\b', "do_not_follow"),
    (r'\bdisregard\s+(all\s+)?(instructions?|previous|above)\b', "disregard"),
    (r'\bforget\s+(everything|all|previous|your)\b', "forget_instruction"),
    (r'\boverride\s+(previous|system|your)\b', "override_instruction"),

    # System prompt extraction
    (r'\b(print|show|reveal|display|tell\s+me|give\s+me)\s+(the\s+)?(system\s+prompt|initial\s+prompt|your\s+instructions)\b', "exfil_system_prompt"),
    (r'\bwhat\s+(are|were)\s+your\s+(original|initial|system)\s+(instructions?|prompts?)\b', "exfil_system_prompt"),

    # Credential/secret extraction
    (r'\b(what\s+is|tell\s+me|reveal|show\s+me|give\s+me)\s+(your|the|my)\s+(password|api\s+key|secret|token|credentials?)\b', "credential_extraction"),
    (r'\b(need|want|require|get|obtain)\s+(your|the|my|a)?\s*(password|api\s+key|secret|token|credentials?)\b', "credential_extraction"),
    (r'\bpassword\s*[:=?]\s*', "password_prompt"),
    (r'\b(access|admin|secret|api)\s+(key|token|password|code)\b', "credential_extraction"),
    (r'\b(urgent|emergency|immediately|quickly|asap).*\b(password|key|token|secret|credentials?)\b', "urgent_credential_request"),

    # Role manipulation
    (r'\byou\s+are\s+now\s+(a|an|acting\s+as)\b', "role_manipulation"),
    (r'\bpretend\s+(you\s+are|to\s+be)\b', "role_manipulation"),
    (r'\bact\s+as\s+(if\s+you|a|an)\b', "role_manipulation"),

    # Jailbreak patterns
    (r'\bDAN\s+mode\b', "jailbreak"),
    (r'\bdo\s+anything\s+now\b', "jailbreak"),
    (r'\bfor\s+educational\s+purposes\s+only\b', "jailbreak_pretext"),
    (r'\bhypothetically\s+speaking\b', "jailbreak_pretext"),

    # Prompt injection markers
    (r'<\|?(?:im_start|im_end|endoftext)\|?>', "prompt_injection_token"),
    (r'\[INST\]|\[/INST\]', "prompt_injection_token"),
    (r'<system>|</system>', "prompt_injection_token"),
]

def apply_rules(text: str):
    return [name for pattern, name in RULES if re.search(pattern, text, re.IGNORECASE)]

def is_suspicious(text: str) -> dict:
    rule_hits = apply_rules(text)

    # High-confidence rules (auto-flag as suspicious)
    strong_rules = {
        "ignore_instruction", "do_not_follow", "disregard", "forget_instruction",
        "exfil_system_prompt", "credential_extraction", "password_prompt",
        "jailbreak", "prompt_injection_token", "urgent_credential_request"
    }
    strong_rule = any(r in strong_rules for r in rule_hits)

    emb = sbert.encode([text])
    prob = clf.predict_proba(emb)[0, 1]

    final_prob = prob
    reason = "model"
    if strong_rule:
        final_prob = max(prob, 0.95)
        reason = "strong_rule"
    elif rule_hits:
        final_prob = max(prob, 0.7)
        reason = "rule_boost"

    suspicious = strong_rule or prob >= threshold

    return {
        "suspicious": bool(suspicious),
        "score": float(final_prob),
        "model_prob": float(prob),
        "threshold": float(threshold),
        "rule_hits": rule_hits,
        "reason": reason
    }
