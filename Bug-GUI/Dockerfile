FROM python:3.12

WORKDIR /bugpredictorr

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["gunicorn","-b","0.0.0.0:8080","app:app","--workers","1","--threads","8"]
