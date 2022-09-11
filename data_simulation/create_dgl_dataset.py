import argparse
import numpy as np
import glob
from pathlib import Path

YAML_STRING = """
dataset_name: multi_dataset
edge_data:
- file_name: edges.csv
node_data:
- file_name: nodes.csv
graph_data:
  file_name: graphs.csv
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_dir")
    parser.add_argument("--outdir")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    df_dir = Path(args.df_dir)

    charges_files = tuple(map(lambda x: df_dir / x, sorted(glob.glob(args.df_dir + "/*_charges.csv"))))
    coordinates_files = tuple(map(lambda x: df_dir / x, sorted(glob.glob(args.df_dir + "/*_coordinates.csv"))))

    coords_charges = zip(coordinates_files, charges_files)
    labels_file = df_dir / "Y.csv"
    n_graphs = len(charges_files)


    outdir.mkdir(exist_ok=True, parents=True)
    edges_fp = outdir / "edges.csv"
    nodes_fp = outdir / "nodes.csv"
    graph_data = outdir / "graphs.csv"

