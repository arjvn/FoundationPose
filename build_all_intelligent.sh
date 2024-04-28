#!/bin/bash
# build_all.sh

# Navigate to the script directory
DIR=$(dirname "$(readlink -f "$0")")

# Function to check if a directory is empty
is_dir_empty() {
  if [ -z "$(ls -A "$1")" ]; then
    return 0 # True, directory is empty
  else
    return 1 # False, directory is not empty
  fi
}

# Build mycpp only if the build directory is not properly set up
if [ ! -f "$DIR/mycpp/build/some_output_file" ]; then
  echo "Building mycpp..."
  cd "$DIR/mycpp/"
  mkdir -p build && cd build
  cmake .. && make -j$(nproc)
else
  echo "Skipping mycpp build; already built."
fi

# Install Kaolin only if it is not already installed
if ! python -c "import kaolin" &> /dev/null; then
  echo "Installing Kaolin..."
  cd /kaolin
  rm -rf build *egg*
  pip install -e .
else
  echo "Kaolin already installed; skipping."
fi

# Install MyCUDA from bundlesdf only if not already installed
if ! python -c "import mycuda" &> /dev/null; then
  echo "Installing MyCUDA..."
  cd "$DIR/bundlesdf/mycuda"
  rm -rf build *egg*
  pip install -e .
else
  echo "MyCUDA already installed; skipping."
fi

# Return to the original directory
cd $DIR
