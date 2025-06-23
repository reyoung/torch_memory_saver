#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the path to torch_memory_saver_cpp.abi3.so
# Assuming it's in the build/lib.linux-x86_64-cpython-312/ directory
SO_PATH="$SCRIPT_DIR/../build/lib.linux-x86_64-cpython-312/torch_memory_saver_cpp.abi3.so"

# Check if the .so file exists
if [ ! -f "$SO_PATH" ]; then
    echo "Error: $SO_PATH not found!"
    echo "Please make sure torch_memory_saver is built correctly."
    exit 1
fi

echo "Using LD_PRELOAD=$SO_PATH"
echo "Running rl_example.py..."

# Set LD_PRELOAD environment variable and run the Python script
LD_PRELOAD="$SO_PATH" python "$SCRIPT_DIR/rl_example.py" 