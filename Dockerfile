FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

# Update container
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y  # Required for CV2

# Set working dir to default notebook dir
WORKDIR /tf/notebooks/

# Copy and install requirements
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r ./requirements.txt