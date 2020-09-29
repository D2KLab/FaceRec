FROM python:3.7-slim-stretch

LABEL Author="Pasquale Lisena"
LABEL E-mail="pasquale.lisena@eurecom.fr"
LABEL version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE 1
ENV FLASK_APP "server.py"
ENV FLASK_ENV "production"
ENV FLASK_DEBUG True

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

RUN apt-get -y update
RUN apt-get install -y --fix-missing build-essential cmake  && apt-get clean && rm -rf /tmp/* /var/tmp/*


RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY mtcnn_patch.sh /app/
RUN app/mtcnn_patch.sh

COPY icrawler_patch.sh /app/
RUN app/icrawler_patch.sh

COPY . /app

EXPOSE 5000

CMD flask run --host=0.0.0.0
