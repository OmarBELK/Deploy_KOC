FROM python:3.7

LABEL maintainer="GenericMlApp"


COPY ./requirements.txt /requirements.txt
COPY ./app /app
COPY ./scripts /scripts


ENV PYTHONBUFFERED=1

WORKDIR /app
EXPOSE 8000


RUN apt-get update &&\
    apt-get install -y --no-install-recommends postgresql-client &&\
    rm -rf /var/lib/apt/lists/*
    

RUN python -m venv /py  &&\
    /py/bin/pip install --upgrade pip &&\
    /py/bin/pip install --default-timeout=100  tensorflow=="2.5.0"  &&\
    /py/bin/pip install -r /requirements.txt  &&\
    adduser --disabled-password --no-create-home app &&\
    mkdir -p /vol/web/static &&\
    mkdir -p /vol/web/media &&\
    chown -R app:app /vol &&\
    chmod -R 755 /vol &&\
    chmod -R +x /scripts


ENV PATH="/scripts:/py/bin:$PATH"

USER root

CMD [ "run.sh" ]