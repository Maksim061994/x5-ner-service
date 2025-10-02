FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH=$PATH:/home/x5/.local/bin

WORKDIR /opt/
COPY . /opt/

RUN apt-get update && \
    apt-get -y install apt-utils tzdata locales nano curl telnet gcc g++ && \
    apt-get clean && apt-get autoclean && apt-get autoremove && rm -rf /var/lib/apt/lists/* && \
    ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
	groupadd -g 5000 x5 && \
    useradd -u 5000 -g x5 -s /bin/bash -m x5 && \
	chown -R x5:x5 /opt

ENV TZ=Europe/Moscow
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER x5
RUN pip install --no-cache-dir -r requirements.txt
	
#No start app
CMD ["tail", "-f", "/dev/null"]