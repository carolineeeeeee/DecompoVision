FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive \
	apt-get install -y libmagickwand-dev ffmpeg libsm6 libxext6
COPY requirements.Dockerfile.txt requirements.Dockerfile.txt
RUN pip install -r requirements.Dockerfile.txt && rm requirements.Dockerfile.txt
WORKDIR /checking
COPY ./checking .
RUN cd detectron2 && python setup.py develop 

# docker build . -t anonymoresearcher/reliability-object-detection:latest

# docker run -it --rm --gpus all -v $HOME/datasets/:/root/datasets anonymoresearcher/reliability-object-detection:latest
