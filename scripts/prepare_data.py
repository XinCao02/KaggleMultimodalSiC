import argparse
from citebench.data import prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="data/processed")
    a = p.parse_args()
    prepare(a.data_dir, a.out_dir)
