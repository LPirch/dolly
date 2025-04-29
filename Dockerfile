FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV PYTHON_VERSION 3.11.9
ENV JAVA_VERSION "17.0.9-tem"
ENV JOERN_VERSION "v4.0.49"
ENV JOERN_HOME="/opt/joern"
ENV PATH="${PATH}:${JOERN_HOME}/joern-cli"
ENV PYTHONUNBUFFERED=1
ENV LC_ALL C.UTF-8
ENV JAVA_OPTS "-Xmx20G"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

ENV DEV_DEPS="git locales time gfortran vim libpq-dev"
ENV PYTHON_DEPS="wget tar gradle software-properties-common build-essential libgraphviz-dev tar zlib1g-dev libffi-dev libreadline-dev liblzma-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev p7zip-full git"
ENV JOERN_DEPS="curl unzip zip"


COPY scripts/docker-setup /app/setup

RUN /app/setup/install_deps.sh $DEV_DEPS $PYTHON_DEPS $JOERN_DEPS && \
    /app/setup/install_python.sh $PYTHON_VERSION && \
    /app/setup/install_java.sh $JAVA_VERSION && \
    /app/setup/install_joern.sh $JOERN_VERSION $JOERN_HOME && \
    rm -rf /app/setup

WORKDIR /app/dolly

COPY . .

RUN python -m venv /root/.venv && \
    source /root/.venv/bin/activate && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r extra-requirements.txt

# download the pre-trained language models
RUN source /root/.venv/bin/activate && \
    dolly init-hf-cache
