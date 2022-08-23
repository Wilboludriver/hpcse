import numpy as np
import os
from pathlib import Path
import random
import shutil
import sys

def run_test(name, files, eps=1e-6):
    
    # Compare files
    count = 0
    for ref_file, test_file in files:
        ref_vec = np.loadtxt(ref_file, delimiter=",").flatten()
        test_vec = np.loadtxt(test_file, delimiter=",").flatten()
        count += 1 if np.linalg.norm(ref_vec - test_vec) < eps else 0

    # Print results
    print("{} - {}: {} / {} subtests passed".format(
        name,
        "passed" if count == len(files) else "failed",
        count,
        len(files)
    ))


def gen_test_data(out_dir, seed, n_test = 10):
    
    # Seed RNG
    random.seed(seed)

    # Generate output
    for _ in range(n_test):
        N = random.randint(1, 1000)
        D = random.randint(1, N)
        k = random.randint(1, D)
        s = random.randint(1, 100)
        os.system("./test {} {} {} {} {}".format(N, D, k, s, out_dir))


if __name__ == "__main__":

    # Set up directory to write files
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True, parents=True)

    # Get seed from input - assuming that the reference data exists
    # in the folder ./test{seed}/
    seed = int(sys.argv[1])
    base_dir = Path("test{}".format(seed))
    gen_test_data(test_dir, seed)

    # Run forward test
    for name in ["forward", "backward"]:
        files = list(zip(base_dir.glob("**/{}*".format(name)), test_dir.glob("**/{}*".format(name))))
        run_test(name[0].upper() + name[1:], files)

    # Clean up
    shutil.rmtree(test_dir)
