import sys
from pathlib import Path

# Make `chemverify` importable under `python -I` (isolated mode ignores PYTHONPATH).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
