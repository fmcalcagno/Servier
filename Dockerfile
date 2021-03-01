FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

RUN pip install torch

RUN conda install -c conda-forge rdkit
RUN conda install -c anaconda networkx

COPY . /workspace/servier/

WORKDIR /workspace/servier/

RUN python setup.py build
RUN python setup.py install
RUN pip install flask
EXPOSE 7000



