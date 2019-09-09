FROM cympfh/python-cuda:3.6.9-9.0

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
