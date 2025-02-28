FROM python:3.12

# Çalışma dizinini ayarla
WORKDIR /bugpredictorr

# Bağımlılıkları yükle
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .
EXPOSE 8080
# Uygulamayı gunicorn ile başlat, worker sayısını CPU çekirdek sayısına göre ayarla
CMD ["gunicorn","-b","0.0.0.0:8080","app:app","--workers","1","--threads","8"]
