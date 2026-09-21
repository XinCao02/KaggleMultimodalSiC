"""Export the CITE portion of evaluation_ids; this is NOT a full Kaggle submission."""

import argparse
import gzip
import numpy as np
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="results/final/cite_predictions.npz")
    p.add_argument("--evaluation-ids", required=True)
    p.add_argument("--output", default="results/final/cite_evaluation_rows.csv.gz")
    a = p.parse_args()
    data = np.load(a.predictions)
    cells = pd.Index(data["cell_ids"])
    proteins = pd.Index(data["proteins"])
    values = data["predictions"]
    count = 0
    with gzip.open(a.output, "wt") as f:
        f.write("row_id,target\n")
        for block in pd.read_csv(a.evaluation_ids, chunksize=250000):
            ix = cells.get_indexer(block.cell_id)
            mask = ix >= 0
            part = block.loc[mask]
            jx = proteins.get_indexer(part.gene_id)
            if (jx < 0).any():
                raise ValueError("Unrecognized CITE target")
            result = pd.DataFrame(
                {"row_id": part.row_id.values, "target": values[ix[mask], jx]}
            )
            result.to_csv(f, index=False, header=False)
            count += len(result)
    if count != values.size:
        raise ValueError(f"Expected {values.size} CITE labels, found {count}")
    print(
        "Exported",
        count,
        "CITE rows. Supply separately computed Multiome rows for a complete competition submission.",
    )
