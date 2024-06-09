FROM bitnami/spark:latest

USER root

COPY .. /app/.

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["spark-submit", "src/kmeans.py"]