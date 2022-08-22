FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive \
	apt-get install -y libmagickwand-dev ffmpeg libsm6 libxext6
COPY requirements.Dockerfile.txt requirements.Dockerfile.txt
RUN pip install -r requirements.Dockerfile.txt && rm requirements.Dockerfile.txt

# docker build . -t huakunshen/reliability-object-detection:latest