'''
Used to print the structure of HDF5 files.
Usage: python print_h5_structure.py file1.h5 file2.h5 ...
'''

import argparse
import h5py as h5

from liblattice.preprocess.print_h5 import print_h5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the structure of HDF5 files.")
    parser.add_argument(
        "files", metavar="F", type=str, nargs="+", help="an HDF5 file path"
    )
    args = parser.parse_args()

    for file in args.files:
        try:
            print(f"{file}:")
            with h5.File(file, "r") as f:
                print_h5(f, 1)
        except:
            print("\tERROR")