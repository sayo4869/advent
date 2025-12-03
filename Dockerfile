# ベースイメージ（Python 3.11 slim）
FROM python:3.11-slim

# 環境変数
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 作業ディレクトリ
WORKDIR /app

# システム依存パッケージ（LightGBMのビルドに必要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 必要最小限のライブラリ（軽量化）
COPY requirements-aws.txt .
RUN pip install --no-cache-dir -r requirements-aws.txt

# アプリケーションコード
COPY 09_lightgbm_batch_aws.py .

# ヘルスチェック用
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import lightgbm; print('OK')" || exit 1

# デフォルトコマンド
CMD ["python", "09_lightgbm_batch_aws.py"]
