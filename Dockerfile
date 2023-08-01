FROM continuumio/miniconda3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

# Create the environment
COPY . .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "hiercadnet", "/bin/bash", "-c"]
