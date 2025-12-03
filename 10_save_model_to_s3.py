"""
学習済みLightGBMモデルをS3に保存するスクリプト
〜本番運用の準備〜

Usage:
    python 10_save_model_to_s3.py
    python 10_save_model_to_s3.py --bucket my-bucket --key models/model.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import argparse
import boto3
from io import BytesIO
from typing import List


def create_lag_features(df: pd.DataFrame, lag_days: List[int] = None) -> pd.DataFrame:
    """ラグ特徴量を作成"""
    if lag_days is None:
        lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

    df = df.copy()
    for lag in lag_days:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
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


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """特徴量カラムを取得"""
    exclude_cols = ['date', 'sales', 'event', 'sales_ma7', 'sales_ma30', 'dow_name']
    return [col for col in df.columns if col not in exclude_cols]


def train_model(df: pd.DataFrame):
    """モデルを学習"""
    print("🔧 Preparing features...")
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    feature_cols = get_feature_columns(df)
    print(f"   Features: {len(feature_cols)}")

    # 学習データ（最後の60日を除く）
    train_df = df.iloc[:-60].dropna(subset=feature_cols + ['sales'])
    X_train = train_df[feature_cols]
    y_train = train_df['sales']

    print(f"   Training samples: {len(X_train)}")

    # モデル学習
    print("🌲 Training LightGBM...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 300,
        'verbose': -1,
        'random_state': 42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    print("   ✅ Training completed")

    return model, feature_cols


def save_to_local(model, feature_cols, path: str):
    """ローカルに保存"""
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
    }

    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"💾 Saved model to {path}")


def save_to_s3(model, feature_cols, bucket: str, key: str):
    """S3に保存"""
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
    }

    buffer = BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)

    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )

    print(f"💾 Saved model to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description='Save LightGBM model')
    parser.add_argument('--data', '-d', default='retail_sales_preprocessed.csv', help='Input data path')
    parser.add_argument('--local', '-l', default='lightgbm_model.pkl', help='Local output path')
    parser.add_argument('--bucket', '-b', default=None, help='S3 bucket (optional)')
    parser.add_argument('--key', '-k', default='models/lightgbm_model.pkl', help='S3 key')

    args = parser.parse_args()

    # データ読み込み
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df['date'] = pd.to_datetime(df['date'])

    # モデル学習
    model, feature_cols = train_model(df)

    # ローカル保存
    save_to_local(model, feature_cols, args.local)

    # S3保存（オプション）
    if args.bucket:
        save_to_s3(model, feature_cols, args.bucket, args.key)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
