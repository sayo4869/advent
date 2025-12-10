"""
Chronos-Bolt による小売売上の時系列予測
〜高速版Transformerベースの時系列基盤モデル〜

Chronos-Bolt: Amazon が開発した高速版時系列予測モデル
- 従来のChronosより最大250倍高速
- T5アーキテクチャベース（最適化済み）
- ゼロショット予測が可能
- 特徴量エンジニアリング不要

モデルサイズ比較:
- tiny:  9M params（超軽量、超高速）
- mini:  21M params（軽量、高速）
- small: 48M params（バランス良い）
- base:  205M params（高精度）

必要なライブラリ:
    pip install chronos-forecasting>=1.4.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import japanize_matplotlib
import time
import gc

warnings.filterwarnings('ignore')

# chronos-forecasting 2.x 対応
from chronos import BaseChronosPipeline

# 比較するモデルサイズ
MODEL_SIZES = ["tiny", "mini", "small", "base"]

# モデルのパラメータ数（参考）
MODEL_PARAMS = {
    "tiny": "9M",
    "mini": "21M",
    "small": "48M",
    "base": "205M"
}


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_device() -> str:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_chronos_bolt_model(model_size: str, device: str):
    """
    Chronos-Boltモデルをロード

    Parameters
    ----------
    model_size : str
        モデルサイズ: "tiny", "mini", "small", "base"
    device : str
        デバイス: "cuda", "mps", "cpu"

    Returns
    -------
    BaseChronosPipeline
        ロード済みモデル
    """
    model_name = f"amazon/chronos-bolt-{model_size}"

    print(f"\n⚡ Chronos-Bolt-{model_size} ({MODEL_PARAMS[model_size]}) をロード中...")

    # chronos-forecasting 2.x ではBaseChronosPipelineで統一
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    print(f"   ✅ ロード完了！")
    return pipeline


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    データを学習・テストに分割

    Chronos-Boltは特徴量エンジニアリング不要！
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


def predict_with_chronos_bolt(
    pipeline,
    train_df: pd.DataFrame,
    prediction_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronos-Boltで予測

    Chronos-Boltは分位点（quantiles）を直接出力する決定論的モデル。
    chronos-forecasting 2.x API対応

    デフォルトの分位点: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        予測値（中央値）、下限（10%）、上限（90%）
    """
    # 時系列データをtensorに変換
    context = torch.tensor(train_df['sales'].values, dtype=torch.float32)

    # chronos-forecasting 2.x: predict() はデフォルトの分位点で出力
    # デフォルト: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
    )

    # numpy配列に変換
    forecast_np = forecast.numpy()

    # 形状: (batch, quantiles, horizon) -> squeeze batch
    if forecast_np.ndim == 3:
        forecast_np = forecast_np.squeeze(0)  # (quantiles, horizon)

    # デフォルト分位点 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # index: 0=10%, 4=50%(中央値), 8=90%
    lower = forecast_np[0]   # 10%
    median = forecast_np[4]  # 50%（中央値）
    upper = forecast_np[8]   # 90%

    return median, lower, upper


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """モデルを評価"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def run_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    全モデルサイズで予測を実行

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        各モデルの予測結果、メトリクスの比較表
    """
    device = get_device()
    print(f"📱 デバイス: {device}")

    all_predictions = {}
    all_metrics = []

    for model_size in MODEL_SIZES:
        print("\n" + "=" * 50)
        print(f"⚡ Chronos-Bolt-{model_size} ({MODEL_PARAMS[model_size]})")
        print("=" * 50)

        start_time = time.time()

        # モデルロード
        pipeline = load_chronos_bolt_model(model_size, device)

        # 予測
        print(f"   🔮 {len(test_df)}日間の予測を実行中...")
        predictions, lower, upper = predict_with_chronos_bolt(
            pipeline,
            train_df,
            prediction_length=len(test_df)
        )

        elapsed = time.time() - start_time

        # 評価
        metrics = evaluate_model(test_df['sales'].values, predictions)
        metrics['model'] = f"Bolt-{model_size}"
        metrics['params'] = MODEL_PARAMS[model_size]
        metrics['time_sec'] = round(elapsed, 1)

        print(f"\n   📈 評価結果:")
        print(f"      RMSE: ¥{metrics['RMSE']:,.0f}")
        print(f"      MAE:  ¥{metrics['MAE']:,.0f}")
        print(f"      MAPE: {metrics['MAPE']:.2f}%")
        print(f"      R²:   {metrics['R2']:.4f}")
        print(f"      ⏱️ 実行時間: {elapsed:.1f}秒")

        # 結果を保存
        results = test_df[['date', 'sales']].copy()
        results['prediction'] = predictions
        results['lower'] = lower
        results['upper'] = upper
        all_predictions[model_size] = results

        all_metrics.append(metrics)

        # メモリ解放
        del pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[['model', 'params', 'RMSE', 'MAE', 'MAPE', 'R2', 'time_sec']]

    return all_predictions, metrics_df


def plot_all_models_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_predictions: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    save_path: str = "figures/"
) -> None:
    """全モデルの比較結果をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    colors = {
        'tiny': '#ff6b6b',
        'mini': '#feca57',
        'small': '#48dbfb',
        'base': '#5f27cd'
    }

    # === 1. 時系列比較（全モデル） ===
    fig, ax = plt.subplots(figsize=(14, 7))

    # 実績
    ax.plot(test_df['date'], test_df['sales'],
            label='実績', linewidth=2.5, color='black', marker='o', markersize=3)

    # 各モデルの予測
    for model_size, results in all_predictions.items():
        ax.plot(results['date'], results['prediction'],
                label=f'Bolt-{model_size} ({MODEL_PARAMS[model_size]})',
                linewidth=2, linestyle='--', color=colors[model_size])

    ax.set_title('Chronos-Bolt モデルサイズ別 予測比較', fontsize=14, fontweight='bold')
    ax.set_xlabel('日付')
    ax.set_ylabel('売上（円）')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}22_bolt_all_sizes_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 2. 評価指標の比較 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_list = ['RMSE', 'MAE', 'MAPE', 'R2']
    bar_colors = [colors[s] for s in MODEL_SIZES]

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 2, idx % 2]
        values = metrics_df[metric].values
        models = [f"{s}\n({MODEL_PARAMS[s]})" for s in MODEL_SIZES]

        bars = ax.bar(models, values, color=bar_colors)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, values):
            if metric in ['RMSE', 'MAE']:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'¥{val:,.0f}', ha='center', va='bottom', fontsize=9)
            elif metric == 'MAPE':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        if metric == 'R2':
            ax.set_ylabel('スコア（高いほど良い）')
        else:
            ax.set_ylabel('誤差（低いほど良い）')

    plt.tight_layout()
    plt.savefig(f"{save_path}22_bolt_all_sizes_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 3. 精度 vs 実行時間のトレードオフ ===
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_size in MODEL_SIZES:
        row = metrics_df[metrics_df['model'] == f"Bolt-{model_size}"].iloc[0]
        ax.scatter(row['time_sec'], row['R2'],
                   s=200, color=colors[model_size],
                   label=f'{model_size} ({MODEL_PARAMS[model_size]})', zorder=5)
        ax.annotate(model_size, (row['time_sec'], row['R2']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('実行時間（秒）')
    ax.set_ylabel('R²スコア')
    ax.set_title('Chronos-Bolt: 精度 vs 実行時間 トレードオフ', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}22_bolt_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 4. 各モデルの予測区間比較 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, model_size in enumerate(MODEL_SIZES):
        ax = axes[idx // 2, idx % 2]
        results = all_predictions[model_size]

        ax.plot(results['date'], test_df['sales'],
                label='実績', linewidth=2, color='black')
        ax.plot(results['date'], results['prediction'],
                label='予測', linewidth=2, linestyle='--', color=colors[model_size])
        ax.fill_between(
            results['date'], results['lower'], results['upper'],
            alpha=0.3, color=colors[model_size], label='95%予測区間'
        )

        row = metrics_df[metrics_df['model'] == f"Bolt-{model_size}"].iloc[0]
        ax.set_title(f"Bolt-{model_size} ({MODEL_PARAMS[model_size]}) | R²={row['R2']:.4f}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('日付')
        ax.set_ylabel('売上（円）')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}22_bolt_all_sizes_intervals.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 比較グラフを保存しました（22_bolt_*.png）")


def print_summary(metrics_df: pd.DataFrame) -> None:
    """サマリーを表示"""
    print("\n" + "=" * 60)
    print("🏆 Chronos-Bolt モデルサイズ比較 サマリー")
    print("=" * 60)

    print("\n📊 評価結果一覧:")
    print(metrics_df.to_string(index=False))

    # ベストモデル
    best_r2_idx = metrics_df['R2'].idxmax()
    best_model = metrics_df.loc[best_r2_idx, 'model']
    best_r2 = metrics_df.loc[best_r2_idx, 'R2']

    print(f"\n🥇 最高精度: {best_model} (R²={best_r2:.4f})")

    # 最速モデル
    fastest_idx = metrics_df['time_sec'].idxmin()
    fastest_model = metrics_df.loc[fastest_idx, 'model']
    fastest_time = metrics_df.loc[fastest_idx, 'time_sec']

    print(f"⚡ 最速: {fastest_model} ({fastest_time}秒)")

    # コスパ（R² / 実行時間）
    metrics_df['efficiency'] = metrics_df['R2'] / metrics_df['time_sec']
    best_eff_idx = metrics_df['efficiency'].idxmax()
    best_eff_model = metrics_df.loc[best_eff_idx, 'model']

    print(f"💰 コスパ最良: {best_eff_model}")

    print("\n" + "-" * 40)
    print("📝 Chronos-Bolt モデル選択の指針:")
    print("-" * 40)
    print("""
⚡ Chronos-Bolt は従来Chronosより最大250倍高速！

・tiny:  超高速処理、エッジデバイス向け
・mini:  軽量かつ実用的、リアルタイム処理
・small: バランス最良、日次バッチ処理
・base:  最高精度、重要な予測タスク

💡 従来Chronosとの使い分け:
  - リアルタイム性重視 → Bolt
  - 精度最優先 → 従来Chronos-large
  - バッチ処理 → Bolt-small or Bolt-base
""")


def main():
    """メイン処理"""
    print("=" * 60)
    print("⚡ Chronos-Bolt モデルサイズ比較")
    print("   tiny / mini / small / base")
    print("=" * 60)

    # データ読み込み
    df = pd.read_csv("retail_sales_preprocessed.csv")

    # データ分割
    train_df, test_df = prepare_data(df)

    print(f"\n📅 データ分割:")
    print(f"   学習データ: {train_df['date'].min()} 〜 {train_df['date'].max()} ({len(train_df)}件)")
    print(f"   テストデータ: {test_df['date'].min()} 〜 {test_df['date'].max()} ({len(test_df)}件)")

    # 全モデルで予測
    all_predictions, metrics_df = run_all_models(train_df, test_df)

    # プロット
    plot_all_models_comparison(train_df, test_df, all_predictions, metrics_df)

    # サマリー表示
    print_summary(metrics_df)

    # 結果を保存
    # ベストモデル（R²最高）の予測結果を標準出力として保存
    best_size = metrics_df.loc[metrics_df['R2'].idxmax(), 'model'].replace('Bolt-', '')
    best_results = all_predictions[best_size]
    best_results.to_csv("chronos_bolt_predictions.csv", index=False)
    print(f"\n✅ ベストモデル({best_size})の予測結果を chronos_bolt_predictions.csv に保存")

    # 全サイズの予測結果を保存
    for model_size, results in all_predictions.items():
        results.to_csv(f"chronos_bolt_predictions_{model_size}.csv", index=False)
    print("✅ 各サイズの予測結果を chronos_bolt_predictions_*.csv に保存")

    # メトリクス比較を保存
    metrics_df.to_csv("chronos_bolt_size_comparison.csv", index=False)
    print("✅ サイズ比較結果を chronos_bolt_size_comparison.csv に保存")

    return all_predictions, metrics_df


if __name__ == "__main__":
    all_predictions, metrics_df = main()
