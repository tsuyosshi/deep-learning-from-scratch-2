FROM python:3

RUN apt-get update
RUN pip install --upgrade pip
RUN python -m pip install jupyterlab

COPY ./opt/ /root/opt/

WORKDIR /root/opt/
COPY Pipfile Pipfile.lock /root/opt/
RUN pip install pipenv
RUN pipenv install --system

