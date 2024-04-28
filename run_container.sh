#!/bin/bash

TEST_SCENE_DIR=${1:-default_test_scene_dir}

# Remove existing Docker container
docker rm -f foundationpose

# Get current directory
DIR=$(pwd)

# Enable X11 forwarding
xhost +

# Define the Google Drive URL and file ID for the dataset
GDRIVE_URL="https://drive.google.com/uc?id=1WgyOXzUa_zOHbtz_e-dZDVLymAqkJqlQ"
DATASET_NAME="avocado_dataset.zip"
TARGET_DIR="$DIR/test_data"
CHECK_TEST_SCENE_DIR="$DIR/test_data/avocado_translate_1/JPEGImages" # Ensuring full path is correct

# Define IDs for model weights
WEIGHT_ID1="1rOoyVobghEQuUfOJmHj8gjZ-UwwzNDk_"
WEIGHT_ID2="1uh6OETH9k-vIiqOxIkQy9muHXKi7U98o"
WEIGHTS_DIR="$DIR/weights"

# Run Docker container and execute the script
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt \
-v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE \
foundationpose:latest bash -c "\
echo 'Requested test scene directory: ${TEST_SCENE_DIR}'; \
if [ '${TEST_SCENE_DIR}' == 'test_data/avocado_translate_1' ]; then \
    if [ -z \"\$(ls -A $CHECK_TEST_SCENE_DIR)\" ]; then \
        echo 'Required test scene directory is avocado_translate_1, checking dataset...'; \
        mkdir -p ${TARGET_DIR}; \
        echo 'Downloading and preparing the dataset...'; \
        pip install gdown; \
        gdown ${GDRIVE_URL} -O ${TARGET_DIR}/${DATASET_NAME} && \
        unzip -q ${TARGET_DIR}/${DATASET_NAME} -d ${TARGET_DIR} && \
        rm ${TARGET_DIR}/${DATASET_NAME}; \
    else \
        echo -e '\033[32m   >>Directory is not empty. Skipping unzip to avoid overwriting files.\033[0m'; \
    fi \
fi; \
echo 'Downloading model weights...'; \
mkdir -p ${DIR}/weights; \
gdown --id 1rOoyVobghEQuUfOJmHj8gjZ-UwwzNDk_ -O ${DIR}/weights/weight1.zip && \
gdown --id 1uh6OETH9k-vIiqOxIkQy9muHXKi7U98o -O ${DIR}/weights/weight2.zip && \
echo 'Unzipping weights...'; \
unzip -qo ${DIR}/weights/weight1.zip -d ${DIR}/weights/ && \
unzip -qo ${DIR}/weights/weight2.zip -d ${DIR}/weights/ && \
rm ${DIR}/weights/*.zip; \
echo 'Running build_all_intelligent.sh...'; \
bash ${DIR}/build_all_intelligent.sh; \
cd $DIR && python run_avocado.py --test_scene_dir '${TEST_SCENE_DIR}' && exec bash"
