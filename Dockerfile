FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tha && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orcwithapi.ipynb .
COPY orcwithpy.ipynb .

COPY *.png /app/
COPY *.jpg /app/

EXPOSE 8866

CMD ["voila", "--port=8866", "--no-browser", "--enable_nbextensions=True"]