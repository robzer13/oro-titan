import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # dossier racine du projet
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
