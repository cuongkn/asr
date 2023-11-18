ARG IMAGE_NAME=pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM ${IMAGE_NAME}

# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# build-essential is needed to install librosa
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install build-essential -y

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -q torchaudio==0.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

# set PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspace/src"

# Copy the rest of the files
COPY . .

RUN git config --global --add safe.directory /workspace

# Run bash by default
CMD ["bash"]
