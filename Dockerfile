# build arguments
ARG PYTHON=python3.10
ARG USER_NAME=lab
ARG VENV_PATH=/var/opt/venv
ARG PORT=8888

# ---------------------------------------------
# source: ubuntu LTS
FROM ubuntu:20.04 AS source
LABEL stage=builder
# ---------------------------------------------

ARG PYTHON
ARG USER_NAME
ARG VENV_PATH

# cancel user prompts
ARG DEBIAN_FRONTEND=noninteractive

# comment out if using current version
RUN apt-get update && apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa

# install python
RUN apt-get update && apt-get install --no-install-recommends -y \
    $PYTHON $PYTHON-dev $PYTHON-venv python3-pip python3-wheel build-essential \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
RUN $PYTHON -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# install requirements
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir wheel
# install notebooks
RUN python -m pip install --no-cache-dir jupyter
# install torch with GPU (consult https://pytorch.org/get-started/locally/ on proper configuration)
RUN python -m pip install --no-cache-dir "torch>=2.0" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --upgrade
# install other python packages
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------
# target: multi-stage for size and performance
FROM ubuntu:20.04 AS target
# ---------------------------------------------

ARG PYTHON
ARG USER_NAME
ARG VENV_PATH
ARG PORT
ARG DEBIAN_FRONTEND

# same as above
RUN apt-get update && apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:alex-p/tesseract-ocr-devel

# same as above except we do not need -dev here
RUN apt-get update && apt-get install --no-install-recommends -y $PYTHON python3-venv git \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# set GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# create non-root user and set environment
RUN useradd --create-home -s /bin/bash --no-user-group -u 1000 $USER_NAME
COPY --chown=1000 --from=source $VENV_PATH $VENV_PATH

USER $USER_NAME
WORKDIR /home/$USER_NAME/workspace

# messages always reach console
ENV PYTHONUNBUFFERED=1
# no __pycache__
ENV PYTHONDONTWRITEBYTECODE=1

# activate virtual environment
ENV VIRTUAL_ENV="$VENV_PATH" 
ENV PATH="$VENV_PATH/bin:$PATH"

# run jupyter server mounted into notebooks folder
ENTRYPOINT jupyter notebook --no-browser --ip=0.0.0.0 --port=${PORT} \
                            --NotebookApp.max_buffer_size=1e6 \
                            --NotebookApp.iopub_data_rate_limit=1e10 \
                            --NotebookApp.iopub_msg_rate_limit=1e10
