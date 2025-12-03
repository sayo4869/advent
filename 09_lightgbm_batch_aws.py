"""
LightGBM AWS ECS用バッチ予測スクリプト
〜S3からデータ取得 → 特徴量生成 → 予測 → S3に保存〜

環境変数:
    S3_INPUT_BUCKET: 入力データのS3バケット
    S3_INPUT_KEY: 入力データのS3キー
    S3_OUTPUT_BUCKET: 出力先のS3バケット
    S3_OUTPUT_PREFIX: 出力先のS3プレフィックス
    S3_MODEL_BUCKET: モデルファイルのS3バケット
    S3_MODEL_KEY: モデルファイルのS3キー
    PREDICTION_DAYS: 予測日数
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
import lightgbm as lgb
import json
import logging
import boto3
from io import StringIO, BytesIO
import pickle

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMForecastBatchAWS:
    """LightGBM AWS用売上予測バッチ処理クラス"""

    def __init__(self):
        # 環境変数から設定を取得
        self.s3_input_bucket = os.environ.get('S3_INPUT_BUCKET', 'my-forecast-bucket')
        self.s3_input_key = os.environ.get('S3_INPUT_KEY', 'data/retail_sales_preprocessed.csv')
        self.s3_output_bucket = os.environ.get('S3_OUTPUT_BUCKET', 'my-forecast-bucket')
        self.s3_output_prefix = os.environ.get('S3_OUTPUT_PREFIX', 'forecasts/')
        self.s3_model_bucket = os.environ.get('S3_MODEL_BUCKET', 'my-forecast-bucket')
        self.s3_model_key = os.environ.get('S3_MODEL_KEY', 'models/lightgbm_model.pkl')
        self.prediction_days = int(os.environ.get('PREDICTION_DAYS', '30'))

        self.s3_client = boto3.client('s3')
        self.model = None
        self.feature_cols = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_model_from_s3(self):
        """S3からLightGBMモデルをロード"""
        logger.info(f"🤖 Loading LightGBM model from s3://{self.s3_model_bucket}/{self.s3_model_key}...")

        response = self.s3_client.get_object(
            Bucket=self.s3_model_bucket,
            Key=self.s3_model_key
        )

        model_data = pickle.loads(response['Body'].read())
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']

        logger.info(f"   ✅ Model loaded ({len(self.feature_cols)} features)")

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

        if 'sales' not in df.columns:
            raise ValueError("'sales' column not found in data")

        logger.info(f"   ✅ Loaded {len(df)} records")
        logger.info(f"   📅 Latest date: {df['date'].max()}")

        return df

    def create_lag_features(self, df: pd.DataFrame, lag_days: List[int] = None) -> pd.DataFrame:
        """ラグ特徴量を作成"""
        if lag_days is None:
            lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

        df = df.copy()
        for lag in lag_days:
            df[f'lag_{lag}'] = df['sales'].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """ローリング特徴量を作成"""
        if windows is None:
            windows = [7, 14, 28]

        df = df.copy()
        for window in windows:
            shifted = df['sales'].shift(1)
            df[f'rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
            df[f'rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
            df[f'rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()
        return df

    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """日付特徴量を作成"""
        df = df.copy()

        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        df['season'] = df['month'].map({
            1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
            7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0
        })

        # サイン・コサイン変換
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を準備"""
        df = self.create_date_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        return df

    def predict_recursive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        再帰的に予測（1日ずつ予測して特徴量を更新）

        ⚠️ ポイント: 複数日予測する場合、前日の予測値を使って
        次の日の特徴量を作る必要がある
        """
        logger.info(f"🔮 Predicting next {self.prediction_days} days (recursive)...")

        df = df.copy()
        last_date = df['date'].max()
        predictions = []

        for day in range(1, self.prediction_days + 1):
            # 次の日の日付
            next_date = last_date + pd.Timedelta(days=day)

            # 新しい行を追加（salesはNaN）
            new_row = pd.DataFrame({'date': [next_date], 'sales': [np.nan]})
            df = pd.concat([df, new_row], ignore_index=True)

            # 特徴量を再計算
            df = self.prepare_features(df)

            # 最後の行で予測
            X = df[self.feature_cols].iloc[-1:].fillna(0)
            pred = self.model.predict(X)[0]

            # 予測値をsalesに設定（次の日のラグ特徴量に使う）
            df.loc[df.index[-1], 'sales'] = pred

            predictions.append({
                'date': next_date,
                'forecast': int(pred),
            })

            if day % 10 == 0:
                logger.info(f"   ... {day}/{self.prediction_days} days completed")

        results = pd.DataFrame(predictions)
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
            'model': 'LightGBM',
            'prediction_days': self.prediction_days,
            'created_at': datetime.now().isoformat(),
            'forecast_start': results['date'].min().isoformat(),
            'forecast_end': results['date'].max().isoformat(),
            'forecast_mean': float(results['forecast'].mean()),
            'forecast_total': float(results['forecast'].sum()),
            's3_input': f"s3://{self.s3_input_bucket}/{self.s3_input_key}",
            's3_model': f"s3://{self.s3_model_bucket}/{self.s3_model_key}",
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

        # latest.json更新
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
        logger.info("🚀 Starting LightGBM batch forecast job")
        logger.info(f"   Run ID: {self.run_id}")
        logger.info("=" * 60)

        try:
            # モデルロード
            self.load_model_from_s3()

            # データ読み込み
            df = self.load_data_from_s3()

            # 予測（再帰的）
            results = self.predict_recursive(df)

            # S3に保存
            csv_key, meta_key = self.save_results_to_s3(results)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info(f"✅ Batch job completed successfully")
            logger.info(f"⏱️ Elapsed time: {elapsed:.1f} seconds")
            logger.info("=" * 60)

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
            import traceback
            traceback.print_exc()
            return {
                'statusCode': 500,
                'error': str(e),
                'run_id': self.run_id,
            }


def main():
    """メイン処理"""
    batch = LightGBMForecastBatchAWS()
    result = batch.run()

    print(json.dumps(result, indent=2))

    if result.get('statusCode') != 200:
        exit(1)


if __name__ == "__main__":
    main()
