"""
LightGBM による時系列予測
〜ラグ特徴量で過去から未来を学ぶ〜

⚠️ 重要なポイント：
時系列データでGBDTを使う場合、データリークに細心の注意が必要！
未来のデータを使って過去を予測しないようにしよう
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, List
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    lag_days: List[int] = None
) -> pd.DataFrame:
    """
    ラグ特徴量を作成

    ⚠️ ポイント：予測時点で使えるラグだけを使う
    例えば、翌日を予測する場合はlag=1（前日の値）は使えるが、
    lag=0（当日の値）は使えない

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    target_col : str
        ターゲット列名
    lag_days : List[int]
        ラグの日数リスト

    Returns
    -------
    pd.DataFrame
        ラグ特徴量が追加されたデータ
    """
    if lag_days is None:
        # デフォルトのラグ（1日前から7日前、14日前、28日前）
        lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

    df = df.copy()

    print(f"📊 ラグ特徴量を作成中...")

    for lag in lag_days:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        print(f"   - lag_{lag}: {lag}日前の売上")

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    windows: List[int] = None
) -> pd.DataFrame:
    """
    ローリング特徴量（移動統計量）を作成

    ⚠️ ポイント：min_periods を使ってNaNを避けつつ、
    shift(1) で当日のデータを使わないようにする

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    target_col : str
        ターゲット列名
    windows : List[int]
        ウィンドウサイズのリスト

    Returns
    -------
    pd.DataFrame
        ローリング特徴量が追加されたデータ
    """
    if windows is None:
        windows = [7, 14, 28]

    df = df.copy()

    print(f"\n📊 ローリング特徴量を作成中...")

    for window in windows:
        # shift(1) で当日を除外してからrollingを計算
        # これを忘れるとデータリーク！
        shifted = df[target_col].shift(1)

        df[f'rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()

        print(f"   - rolling_{window}: {window}日間の統計量（mean, std, max, min）")

    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    日付に基づく特徴量を作成

    これらは未来のデータを使っていないのでリークの心配なし
    """
    df = df.copy()

    print(f"\n📊 日付特徴量を作成中...")

    # 基本的な日付特徴量（既にあるものはスキップ）
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    if 'day' not in df.columns:
        df['day'] = df['date'].dt.day

    # 追加の日付特徴量
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 月の週（月初、月中、月末）
    df['week_of_month'] = (df['day'] - 1) // 7 + 1

    # 季節（春夏秋冬）
    df['season'] = df['month'].map({
        1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
        7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0
    })

    # サイン・コサイン変換（周期性を捉える）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print("   - day_of_year, week_of_year, week_of_month, season")
    print("   - sin/cos変換（月、曜日）")

    return df


def create_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    イベント特徴量を作成
    """
    df = df.copy()

    print(f"\n📊 イベント特徴量を作成中...")

    # イベントをダミー変数化（ワンホットエンコーディング）
    if 'event' in df.columns:
        event_dummies = pd.get_dummies(df['event'], prefix='event')
        df = pd.concat([df, event_dummies], axis=1)
        print(f"   - イベントをダミー変数化: {list(event_dummies.columns)}")

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ての特徴量を準備

    ⚠️ 重要：ラグ特徴量作成後、NaNが含まれる最初の行は削除が必要
    """
    print("\n" + "=" * 50)
    print("🔧 特徴量エンジニアリング")
    print("=" * 50)

    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # 各特徴量を作成
    df = create_date_features(df)
    df = create_event_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # NaNを含む行数を確認
    nan_rows = df.isnull().any(axis=1).sum()
    print(f"\n⚠️ NaNを含む行数: {nan_rows}")

    return df


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    時系列データを学習・テストに分割

    ⚠️ ポイント：時系列データは絶対にシャッフルしない！
    未来のデータが学習に混じると大惨事
    """
    df = df.sort_values('date').reset_index(drop=True)

    split_idx = len(df) - test_days
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    特徴量として使用するカラムを取得

    除外するもの：
    - date（日付型）
    - sales（ターゲット）
    - event（文字列、ダミー変数化済み）
    - 一時的なカラム
    """
    exclude_cols = ['date', 'sales', 'event', 'sales_ma7', 'sales_ma30', 'dow_name']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def train_lightgbm(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'sales'
) -> lgb.LGBMRegressor:
    """
    LightGBMモデルを学習

    Parameters
    ----------
    train_df : pd.DataFrame
        学習データ
    feature_cols : List[str]
        特徴量カラム
    target_col : str
        ターゲットカラム

    Returns
    -------
    lgb.LGBMRegressor
        学習済みモデル
    """
    print("\n" + "=" * 50)
    print("🌲 LightGBM モデル学習")
    print("=" * 50)

    # NaNを削除
    train_clean = train_df.dropna(subset=feature_cols + [target_col])
    print(f"\n学習データ: {len(train_clean)} 件（NaN削除後）")

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]

    # LightGBMのパラメータ
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 500,
        'early_stopping_rounds': 50,
        'verbose': -1,
        'random_state': 42,
    }

    print(f"\n🔧 パラメータ:")
    for key, value in list(params.items())[:6]:
        print(f"   - {key}: {value}")

    # モデルを学習
    model = lgb.LGBMRegressor(**params)

    # 検証データは学習データの最後の20%を使用
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
    )

    print(f"\n✅ 学習完了！ベスト iteration: {model.best_iteration_}")

    return model


def evaluate_lightgbm(
    model: lgb.LGBMRegressor,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'sales'
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    LightGBMモデルを評価
    """
    print("\n📊 モデルを評価中...")

    # NaNを削除
    test_clean = test_df.dropna(subset=feature_cols + [target_col])
    print(f"テストデータ: {len(test_clean)} 件（NaN削除後）")

    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    # 予測
    y_pred = model.predict(X_test)

    # 評価指標を計算
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    print("\n📈 LightGBM 評価結果:")
    print(f"   RMSE: ¥{metrics['RMSE']:,.0f}")
    print(f"   MAE:  ¥{metrics['MAE']:,.0f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")

    # 結果をDataFrameに
    results = test_clean[['date', target_col]].copy()
    results['prediction'] = y_pred

    return results, metrics


def plot_feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: List[str],
    save_path: str = "figures/"
) -> None:
    """特徴量重要度をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    # 特徴量重要度を取得
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    # 上位20件をプロット
    top_n = 20
    importance_top = importance.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_top['feature'], importance_top['importance'], color='steelblue')
    ax.set_title(f'LightGBM 特徴量重要度（上位{top_n}）', fontsize=14, fontweight='bold')
    ax.set_xlabel('重要度')

    plt.tight_layout()
    plt.savefig(f"{save_path}08_lightgbm_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}08_lightgbm_importance.png を保存しました")


def plot_lightgbm_results(
    results: pd.DataFrame,
    save_path: str = "figures/"
) -> None:
    """予測結果をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(results['date'], results['sales'],
            label='実績', linewidth=2, marker='o', markersize=3)
    ax.plot(results['date'], results['prediction'],
            label='予測', linewidth=2, linestyle='--')

    ax.set_title('テスト期間: 実績 vs 予測 (LightGBM)', fontsize=14, fontweight='bold')
    ax.set_xlabel('日付')
    ax.set_ylabel('売上（円）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}09_lightgbm_test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}09_lightgbm_test_comparison.png を保存しました")


def main():
    """メイン処理"""
    print("=" * 60)
    print("🌲 LightGBM による売上予測")
    print("=" * 60)

    # データ読み込み
    df = pd.read_csv("apparel_sales_preprocessed.csv")
    df['date'] = pd.to_datetime(df['date'])

    # 特徴量を準備
    df = prepare_features(df)

    # 特徴量カラムを取得
    feature_cols = get_feature_columns(df)
    print(f"\n📊 使用する特徴量: {len(feature_cols)} 個")

    # 学習・テストに分割
    train_df, test_df = train_test_split_timeseries(df, test_days=60)
    print(f"\n📅 データ分割:")
    print(f"   学習データ: {train_df['date'].min().strftime('%Y-%m-%d')} 〜 {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)}件)")
    print(f"   テストデータ: {test_df['date'].min().strftime('%Y-%m-%d')} 〜 {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)}件)")

    # モデル学習
    model = train_lightgbm(train_df, feature_cols)

    # 評価
    results, metrics = evaluate_lightgbm(model, test_df, feature_cols)

    # プロット
    plot_feature_importance(model, feature_cols)
    plot_lightgbm_results(results)

    # 結果を保存
    results.to_csv("lightgbm_predictions.csv", index=False)
    print("\n✅ 予測結果を lightgbm_predictions.csv に保存しました")

    # メトリクスを保存
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'LightGBM'
    metrics_df.to_csv("lightgbm_metrics.csv", index=False)

    return model, results, metrics


if __name__ == "__main__":
    model, results, metrics = main()
