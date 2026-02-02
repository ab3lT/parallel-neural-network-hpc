#!/bin/bash
#
# download_mnist.sh - Download and extract MNIST dataset
#
# This script downloads the MNIST handwritten digit dataset from the official source.
# Files are extracted to the ./data/mnist directory.
#
# Usage: ./scripts/download_mnist.sh
#

set -e

# Create data directory
DATA_DIR="data/mnist"
mkdir -p "$DATA_DIR"

# MNIST URLs (from Yann LeCun's website)
BASE_URL="http://yann.lecun.com/exdb/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset..."
echo "Target directory: $DATA_DIR"
echo ""

for file in "${FILES[@]}"; do
    if [ -f "$DATA_DIR/${file%.gz}" ]; then
        echo "  $file already exists (uncompressed), skipping..."
    elif [ -f "$DATA_DIR/$file" ]; then
        echo "  Extracting $file..."
        gunzip -k "$DATA_DIR/$file"
    else
        echo "  Downloading $file..."
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "$DATA_DIR/$file" "$BASE_URL/$file"
        elif command -v curl &> /dev/null; then
            curl -# -o "$DATA_DIR/$file" "$BASE_URL/$file"
        else
            echo "Error: Neither wget nor curl is installed"
            exit 1
        fi
        echo "  Extracting $file..."
        gunzip "$DATA_DIR/$file"
    fi
done

echo ""
echo "MNIST dataset downloaded successfully!"
echo ""
echo "Files in $DATA_DIR:"
ls -la "$DATA_DIR"
echo ""
echo "Usage:"
echo "  ./bin/serial_train --mnist $DATA_DIR -e 20"
echo "  ./bin/openmp_train --mnist $DATA_DIR -e 20 -t 4"
