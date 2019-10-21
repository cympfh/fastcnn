FROM python:3.6.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .

WORKDIR /workdir
