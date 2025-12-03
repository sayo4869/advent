"""
AWS ECS用バッチ予測スクリプト
〜S3からデータ取得 → 予測 → S3に保存〜

環境変数:
    S3_INPUT_BUCKET: 入力データのS3バケット
    S3_INPUT_KEY: 入力データのS3キー
    S3_OUTPUT_BUCKET: 出力先のS3バケット
    S3_OUTPUT_PREFIX: 出力先のS3プレフィックス
    MODEL_SIZE: Chronosモデルサイズ (tiny/mini/small/base/large)
    PREDICTION_DAYS: 予測日数
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
from chronos import ChronosPipeline
import json
import logging
import boto3
from io import StringIO, BytesIO

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesForecastBatchAWS:
    """AWS用売上予測バッチ処理クラス"""

    def __init__(self):
        # 環境変数から設定を取得
        self.s3_input_bucket = os.environ.get('S3_INPUT_BUCKET', 'my-forecast-bucket')
        self.s3_input_key = os.environ.get('S3_INPUT_KEY', 'data/retail_sales_preprocessed.csv')
        self.s3_output_bucket = os.environ.get('S3_OUTPUT_BUCKET', 'my-forecast-bucket')
        self.s3_output_prefix = os.environ.get('S3_OUTPUT_PREFIX', 'forecasts/')
        self.model_size = os.environ.get('MODEL_SIZE', 'small')
        self.prediction_days = int(os.environ.get('PREDICTION_DAYS', '30'))

        self.s3_client = boto3.client('s3')
        self.pipeline = None
        self.device = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_model(self):
        """Chronosモデルをロード"""
        logger.info(f"🤖 Loading Chronos model (size={self.model_size})...")

        # デバイス選択（ECSではCPUが基本、GPU使う場合はcuda）
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model_size}",
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        logger.info(f"   ✅ Model loaded on {self.device}")

    def load_data_from_s3(self) -> pd.DataFrame:
        """S3からデータを読み込み"""
        logger.info(f"📂 Loading data from s3://{self.s3_input_bucket}/{self.s3_input_key}...")

        response = self.s3_client.get_object(
            Bucket=self.s3_input_bucket,
            Key=self.s3_input_key
        )

        df = pd.read_csv(response['Body'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # バリデーション
        if 'sales' not in df.columns:
            raise ValueError("'sales' column not found in data")

        if df['sales'].isnull().any():
            logger.warning("   ⚠️ Found NaN values, filling with forward fill")
            df['sales'] = df['sales'].fillna(method='ffill')

        logger.info(f"   ✅ Loaded {len(df)} records")
        logger.info(f"   📅 Period: {df['date'].min()} ~ {df['date'].max()}")

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測を実行"""
        logger.info(f"🔮 Predicting next {self.prediction_days} days...")

        context = torch.tensor(df['sales'].values, dtype=torch.float32)

        forecast = self.pipeline.predict(
            context,
            prediction_length=self.prediction_days,
            num_samples=20,
        )

        forecast_np = forecast.numpy()

        # 予測日付を生成
        last_date = df['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.prediction_days,
            freq='D'
        )

        # 統計量を計算
        median = np.median(forecast_np, axis=1).squeeze()
        lower_95 = np.percentile(forecast_np, 2.5, axis=1).squeeze()
        upper_95 = np.percentile(forecast_np, 97.5, axis=1).squeeze()

        results = pd.DataFrame({
            'date': forecast_dates,
            'forecast': median.astype(int),
            'lower_95': lower_95.astype(int),
            'upper_95': upper_95.astype(int),
        })

        logger.info("   ✅ Prediction completed")
        logger.info(f"   📊 Forecast mean: ¥{results['forecast'].mean():,.0f}")

        return results

    def save_results_to_s3(self, results: pd.DataFrame):
        """結果をS3に保存"""
        # CSV保存
        csv_key = f"{self.s3_output_prefix}forecast_{self.run_id}.csv"
        csv_buffer = StringIO()
        results.to_csv(csv_buffer, index=False)

        self.s3_client.put_object(
            Bucket=self.s3_output_bucket,
            Key=csv_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        logger.info(f"💾 Saved forecast to s3://{self.s3_output_bucket}/{csv_key}")

        # メタデータ保存
        metadata = {
            'run_id': self.run_id,
            'model': f'chronos-t5-{self.model_size}',
            'device': self.device,
            'prediction_days': self.prediction_days,
            'created_at': datetime.now().isoformat(),
            'forecast_start': results['date'].min().isoformat(),
            'forecast_end': results['date'].max().isoformat(),
            'forecast_mean': float(results['forecast'].mean()),
            'forecast_total': float(results['forecast'].sum()),
            's3_input': f"s3://{self.s3_input_bucket}/{self.s3_input_key}",
            's3_output': f"s3://{self.s3_output_bucket}/{csv_key}",
        }

        meta_key = f"{self.s3_output_prefix}metadata_{self.run_id}.json"
        self.s3_client.put_object(
            Bucket=self.s3_output_bucket,
            Key=meta_key,
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )
        logger.info(f"💾 Saved metadata to s3://{self.s3_output_bucket}/{meta_key}")

        # 最新予測へのポインタを更新（latest.json）
        latest = {
            'latest_run_id': self.run_id,
            'latest_forecast': f"s3://{self.s3_output_bucket}/{csv_key}",
            'updated_at': datetime.now().isoformat(),
        }
        self.s3_client.put_object(
            Bucket=self.s3_output_bucket,
            Key=f"{self.s3_output_prefix}latest.json",
            Body=json.dumps(latest, indent=2),
            ContentType='application/json'
        )

        return csv_key, meta_key

    def run(self) -> dict:
        """バッチ処理を実行"""
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("🚀 Starting AWS batch forecast job")
        logger.info(f"   Run ID: {self.run_id}")
        logger.info("=" * 60)

        try:
            # モデルロード
            self.load_model()

            # データ読み込み
            df = self.load_data_from_s3()

            # 予測
            results = self.predict(df)

            # S3に保存
            csv_key, meta_key = self.save_results_to_s3(results)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info(f"✅ Batch job completed successfully")
            logger.info(f"⏱️ Elapsed time: {elapsed:.1f} seconds")
            logger.info("=" * 60)

            # Step Functions用のレスポンス
            return {
                'statusCode': 200,
                'run_id': self.run_id,
                'forecast_count': len(results),
                'forecast_mean': float(results['forecast'].mean()),
                's3_output': f"s3://{self.s3_output_bucket}/{csv_key}",
                'elapsed_seconds': elapsed,
            }

        except Exception as e:
            logger.error(f"❌ Batch job failed: {e}")
            return {
                'statusCode': 500,
                'error': str(e),
                'run_id': self.run_id,
            }


def main():
    """メイン処理"""
    batch = SalesForecastBatchAWS()
    result = batch.run()

    # 結果を標準出力（Step Functionsで取得可能）
    print(json.dumps(result, indent=2))

    # エラーの場合は終了コード1
    if result.get('statusCode') != 200:
        exit(1)


if __name__ == "__main__":
    main()
