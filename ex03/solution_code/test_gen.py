import os
from pathlib import Path
import sys

from test_run import gen_test_data

if __name__ == "__main__":

    # Get seed from input - create reference data for test_run.py
    seed = int(sys.argv[1])
    base_dir = Path("test{}".format(seed))
    base_dir.mkdir(exist_ok=True, parents=True)
    gen_test_data(base_dir, seed)
