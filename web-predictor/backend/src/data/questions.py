import json
from pathlib import Path

QUESTIONS_FILE = Path(__file__).parent.parent / "static" / "questions.json"

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    QUESTIONS = json.load(f)
