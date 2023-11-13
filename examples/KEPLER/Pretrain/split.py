import random
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--output_dir", type=str, help="directory to save train/valid/test files")

if __name__=='__main__':
    args = parser.parse_args()

    # Set the path to the input file and the names for the output files
    if not os.path.exists(args.input_file):
        print(f"Input file does not exist at {args.input_file}")
        exit(1)

    train_file = os.path.join(args.output_dir, "train.txt")
    val_file = os.path.join(args.output_dir, "valid.txt")
    test_file = os.path.join(args.output_dir, "test.txt")

    # Set the percentage splits
    train_pct = 0.7
    val_pct = 0.2
    test_pct = 0.1

    # Read in the input file
    with open(args.input_file, "r") as f:
        items = f.readlines()

    # Shuffle the items
    random.seed(42)
    random.shuffle(items)

    # Calculate the number of items for each split
    num_items = len(items)
    num_train = int(train_pct * num_items)
    num_val = int(val_pct * num_items)
    num_test = num_items - num_train - num_val

    # Split the items into train, validation, and test sets
    train_items = items[:num_train]
    val_items = items[num_train:num_train+num_val]
    test_items = items[num_train+num_val:]

    # Write the items to the output files
    with open(train_file, "w") as f:
        f.writelines(train_items)

    with open(val_file, "w") as f:
        f.writelines(val_items)

    with open(test_file, "w") as f:
        f.writelines(test_items)
