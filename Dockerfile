# Dockerfile - lean runtime image for the market notifier worker
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy requirements first to leverage layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

EXPOSE 8080

# run the always-on worker which includes a small health endpoint
CMD ["python", "worker.py"]
