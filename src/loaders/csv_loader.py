from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_csv(path: Path) -> List[Dict]:
    df = pd.read_csv(path)
    rows = []
    for i, row in df.iterrows():
        rows.append({"text": row.to_json(), "metadata": {"source": str(path), "row": i}})
    return rows
