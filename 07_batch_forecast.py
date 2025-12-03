"""
バッチ予測スクリプト
〜Chronosを定期実行で本番運用する〜

Usage:
    python 07_batch_forecast.py
    python 07_batch_forecast.py --model-size base --days 60
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
from chronos import ChronosPipeline
import json
import logging
import argparse

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesForecastBatch:
    """売上予測バッチ処理クラス"""

    def __init__(
        self,
        model_size: str = "small",
        prediction_days: int = 30,
        output_dir: str = "forecasts"
    ):
        """
        Parameters
        ----------
        model_size : str
            Chronosモデルサイズ: tiny, mini, small, base, large
        prediction_days : int
            予測日数
        output_dir : str
            出力ディレクトリ
        """
        self.model_size = model_size
        self.prediction_days = prediction_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.pipeline = None
        self.device = None

    def load_model(self):
        """モデルをロード"""
        logger.info(f"🤖 Loading Chronos model (size={self.model_size})...")

        # デバイス選択
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model_size}",
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        logger.info(f"   ✅ Model loaded on {self.device}")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """データを読み込み"""
        logger.info(f"📂 Loading data from {data_path}...")

        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # バリデーション
        if 'sales' not in df.columns:
            raise ValueError("'sales' column not found in data")

        if df['sales'].isnull().any():
            logger.warning("   ⚠️ Found NaN values in sales, filling with forward fill")
            df['sales'] = df['sales'].fillna(method='ffill')

        logger.info(f"   ✅ Loaded {len(df)} records")
        logger.info(f"   📅 Period: {df['date'].min()} ~ {df['date'].max()}")

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測を実行"""
        logger.info(f"🔮 Predicting next {self.prediction_days} days...")

        # tensorに変換
        context = torch.tensor(df['sales'].values, dtype=torch.float32)

        # 予測実行
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
        lower_50 = np.percentile(forecast_np, 25, axis=1).squeeze()
        upper_50 = np.percentile(forecast_np, 75, axis=1).squeeze()

        # 結果をDataFrameに
        results = pd.DataFrame({
            'date': forecast_dates,
            'forecast': median.astype(int),
            'lower_95': lower_95.astype(int),
            'upper_95': upper_95.astype(int),
            'lower_50': lower_50.astype(int),
            'upper_50': upper_50.astype(int),
        })

        logger.info("   ✅ Prediction completed")

        # サマリーを表示
        logger.info(f"   📊 Forecast summary:")
        logger.info(f"      - Mean: ¥{results['forecast'].mean():,.0f}")
        logger.info(f"      - Min:  ¥{results['forecast'].min():,.0f}")
        logger.info(f"      - Max:  ¥{results['forecast'].max():,.0f}")

        return results

    def save_results(self, results: pd.DataFrame, run_id: str = None):
        """結果を保存"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV保存
        csv_path = self.output_dir / f"forecast_{run_id}.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved forecast to {csv_path}")

        # メタデータ保存
        metadata = {
            'run_id': run_id,
            'model': f'chronos-t5-{self.model_size}',
            'device': self.device,
            'prediction_days': self.prediction_days,
            'created_at': datetime.now().isoformat(),
            'forecast_start': results['date'].min().isoformat(),
            'forecast_end': results['date'].max().isoformat(),
            'forecast_mean': float(results['forecast'].mean()),
            'forecast_total': float(results['forecast'].sum()),
        }

        meta_path = self.output_dir / f"metadata_{run_id}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved metadata to {meta_path}")

        return csv_path, meta_path

    def run(self, data_path: str) -> pd.DataFrame:
        """バッチ処理を実行"""
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("🚀 Starting batch forecast job")
        logger.info("=" * 60)

        try:
            # モデルロード
            self.load_model()

            # データ読み込み
            df = self.load_data(data_path)

            # 予測
            results = self.predict(df)

            # 保存
            csv_path, meta_path = self.save_results(results)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info(f"✅ Batch job completed successfully")
            logger.info(f"⏱️ Elapsed time: {elapsed:.1f} seconds")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"❌ Batch job failed: {e}")
            raise


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Sales Forecast Batch Job')
    parser.add_argument(
        '--data', '-d',
        default='retail_sales_preprocessed.csv',
        help='Input data path'
    )
    parser.add_argument(
        '--model-size', '-m',
        default='small',
        choices=['tiny', 'mini', 'small', 'base', 'large'],
        help='Chronos model size'
    )
    parser.add_argument(
        '--days', '-n',
        type=int,
        default=30,
        help='Number of days to predict'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='forecasts',
        help='Output directory'
    )

    args = parser.parse_args()

    batch = SalesForecastBatch(
        model_size=args.model_size,
        prediction_days=args.days,
        output_dir=args.output_dir
    )

    results = batch.run(args.data)

    # 結果をプレビュー
    print("\n📋 Forecast Preview:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
