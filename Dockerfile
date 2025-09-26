# Используем официальный Python образ
FROM python:3.11-slim AS builder

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# set Work Directory
WORKDIR /usr/src/app

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

FROM python:3.11-slim

#Set environment variable
ARG DEBIAN_FRONTEND=noninteractive
#Prevents Python from writing pyc files to disc (equivalent python -B)
ENV PYTHONDONTWRITEBYTECODE 1
#Prevents Python from buffering stdout and stderr (equivalent python -u)
ENV PYTHONUNBUFFERED 1
ENV PATH=$PATH:/home/x5/.local/bin

WORKDIR /opt/

# add app
COPY . /opt/

# install system dependencies and set Moscow time
RUN apt-get update && \
    apt-get -y install apt-utils tzdata locales nano curl telnet && \
    apt-get clean && apt-get autoclean && apt-get autoremove && rm -rf /var/lib/apt/lists/* && \
    ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
	groupadd -g 5000 x5 && \
    useradd -u 5000 -g x5 -s /bin/bash -m x5 && \
	chown -R x5:x5 /opt

#Set locale & timezone environment variable
ENV TZ=Europe/Moscow
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER x5

# install python dependencies
COPY --from=builder --chown=x5:x5 /usr/src/app/wheels /tmp/wheels
RUN pip install --upgrade --no-cache setuptools pip && \
    pip install --no-cache /tmp/wheels/* && \
	rm -rf /tmp/wheels/*