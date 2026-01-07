from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_csv(path: Path) -> List[Dict]:
    """Carga CSV y genera texto legible (columna: valor) con metadata de tipo."""
    df = pd.read_csv(path)
    rows: List[Dict] = []
    for i, row in df.iterrows():
        rendered = "; ".join(f"{col}: {row[col]}" for col in df.columns)
        rows.append(
            {
                "text": rendered,
                "metadata": {"source": str(path), "row": int(i), "type": "csv"},
            }
        )
    return rows
