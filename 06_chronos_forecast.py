"""
Chronos による小売売上の時系列予測
〜Transformerベースの時系列基盤モデル〜

Chronos: Amazon が開発した事前学習済み時系列予測モデル
- T5アーキテクチャベース
- ゼロショット予測が可能
- 特徴量エンジニアリング不要
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_chronos_model(model_size: str = "small") -> ChronosPipeline:
    """
    Chronosモデルをロード

    Parameters
    ----------
    model_size : str
        モデルサイズ: "tiny", "mini", "small", "base", "large"
        - tiny: 8M params（軽量、高速）
        - small: 46M params（バランス良い）
        - base: 200M params（高精度）

    Returns
    -------
    ChronosPipeline
        ロード済みモデル
    """
    model_name = f"amazon/chronos-t5-{model_size}"

    print(f"🤖 Chronosモデルをロード中: {model_name}")
    print("   （初回は数分かかる場合があります）")

    # GPU/MPS/CPUを自動選択
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"   デバイス: {device}")

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    print("   ✅ モデルロード完了！")
    return pipeline


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データを学習・テストに分割

    Chronosは特徴量エンジニアリング不要！
    時系列データをそのまま渡すだけ
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 最後の60日をテストデータに
    test_days = 60
    split_idx = len(df) - test_days

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def predict_with_chronos(
    pipeline: ChronosPipeline,
    train_df: pd.DataFrame,
    prediction_length: int,
    num_samples: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronosで予測

    Parameters
    ----------
    pipeline : ChronosPipeline
        ロード済みモデル
    train_df : pd.DataFrame
        学習データ
    prediction_length : int
        予測期間（日数）
    num_samples : int
        サンプリング数（予測区間用）

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        予測値（中央値）、下限、上限
    """
    print(f"\n🔮 {prediction_length}日間の予測を実行中...")

    # 時系列データをtensorに変換
    context = torch.tensor(train_df['sales'].values, dtype=torch.float32)

    # 予測実行
    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    # numpy配列に変換
    forecast_np = forecast.numpy()

    # 中央値と予測区間を計算
    median = np.median(forecast_np, axis=1).squeeze()
    lower = np.percentile(forecast_np, 2.5, axis=1).squeeze()
    upper = np.percentile(forecast_np, 97.5, axis=1).squeeze()

    print("   ✅ 予測完了！")

    return median, lower, upper


def evaluate_chronos(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Chronosモデルを評価
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

    print("\n📈 Chronos 評価結果:")
    print(f"   RMSE: ¥{metrics['RMSE']:,.0f}")
    print(f"   MAE:  ¥{metrics['MAE']:,.0f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")

    return metrics


def plot_chronos_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    save_path: str = "figures/"
) -> None:
    """結果をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 全期間のプロット
    ax1 = axes[0]

    # 学習データ
    ax1.plot(train_df['date'], train_df['sales'],
             label='学習データ', linewidth=1, alpha=0.7)

    # テストデータ（実績）
    ax1.plot(test_df['date'], test_df['sales'],
             label='実績', linewidth=2, color='black')

    # 予測
    ax1.plot(test_df['date'], predictions,
             label='Chronos予測', linewidth=2, linestyle='--', color='#e74c3c')

    # 予測区間
    ax1.fill_between(
        test_df['date'], lower, upper,
        alpha=0.3, color='#e74c3c', label='95%予測区間'
    )

    ax1.set_title('Chronos（Transformer）による売上予測', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日付')
    ax1.set_ylabel('売上（円）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. テスト期間の拡大
    ax2 = axes[1]

    ax2.plot(test_df['date'], test_df['sales'],
             label='実績', linewidth=2, marker='o', markersize=3, color='black')
    ax2.plot(test_df['date'], predictions,
             label='予測', linewidth=2, linestyle='--', color='#e74c3c')
    ax2.fill_between(
        test_df['date'], lower, upper,
        alpha=0.3, color='#e74c3c', label='95%予測区間'
    )

    ax2.set_title('テスト期間: 実績 vs 予測 (Chronos)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日付')
    ax2.set_ylabel('売上（円）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}14_chronos_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ {save_path}14_chronos_forecast.png を保存しました")


def main():
    """メイン処理"""
    print("=" * 60)
    print("🤖 Chronos（Transformer）による小売売上予測")
    print("=" * 60)

    # データ読み込み
    df = pd.read_csv("retail_sales_preprocessed.csv")

    # データ分割
    train_df, test_df = prepare_data(df)

    print(f"\n📅 データ分割:")
    print(f"   学習データ: {train_df['date'].min()} 〜 {train_df['date'].max()} ({len(train_df)}件)")
    print(f"   テストデータ: {test_df['date'].min()} 〜 {test_df['date'].max()} ({len(test_df)}件)")

    # モデルロード（smallがバランス良い）
    pipeline = load_chronos_model(model_size="small")

    # 予測
    predictions, lower, upper = predict_with_chronos(
        pipeline,
        train_df,
        prediction_length=len(test_df),
        num_samples=20
    )

    # 評価
    metrics = evaluate_chronos(test_df['sales'].values, predictions)

    # プロット
    plot_chronos_results(train_df, test_df, predictions, lower, upper)

    # 結果を保存
    results = test_df[['date', 'sales']].copy()
    results['prediction'] = predictions
    results['lower'] = lower
    results['upper'] = upper
    results.to_csv("chronos_predictions.csv", index=False)
    print("\n✅ 予測結果を chronos_predictions.csv に保存しました")

    # メトリクスを保存
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'Chronos'
    metrics_df.to_csv("chronos_metrics.csv", index=False)

    return pipeline, results, metrics


if __name__ == "__main__":
    pipeline, results, metrics = main()
