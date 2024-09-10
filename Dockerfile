# ベースイメージとしてKaggleのGPU Pythonイメージを指定
FROM gcr.io/kaggle-gpu-images/python

# 作業ディレクトリを設定
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .