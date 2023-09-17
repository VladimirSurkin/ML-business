# запускать сборку через: docker build -t predict_service .
FROM python:3.10
COPY /files/. /app
WORKDIR /app
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]