FROM python:3.7.1

LABEL Author="Pasquale Lisena"
LABEL E-mail="pasquale.lisena@eurecom.fr"
LABEL version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE 1
ENV FLASK_APP "server.py"
ENV FLASK_ENV "production"
ENV FLASK_DEBUG True

RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 5000

CMD flask run --host=0.0.0.0
