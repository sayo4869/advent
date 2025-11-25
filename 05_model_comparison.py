"""
小売売上予測のモデル比較 & 精度評価
〜どのモデルが一番イケてるか決着をつける〜
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_predictions() -> Dict[str, pd.DataFrame]:
    """各モデルの予測結果を読み込む"""
    predictions = {}

    # Prophet
    prophet_df = pd.read_csv("prophet_predictions.csv")
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df.rename(columns={'ds': 'date', 'y': 'actual', 'yhat': 'prediction'})
    predictions['Prophet'] = prophet_df

    # LightGBM
    lgb_df = pd.read_csv("lightgbm_predictions.csv")
    lgb_df['date'] = pd.to_datetime(lgb_df['date'])
    lgb_df = lgb_df.rename(columns={'sales': 'actual'})
    predictions['LightGBM'] = lgb_df

    return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """評価指標を計算"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def compare_models(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """モデル間の比較"""
    print("=" * 60)
    print("📊 モデル比較")
    print("=" * 60)

    results = []

    for model_name, df in predictions.items():
        metrics = calculate_metrics(df['actual'], df['prediction'])
        metrics['model'] = model_name
        results.append(metrics)

        print(f"\n【{model_name}】")
        print(f"   RMSE: ¥{metrics['RMSE']:,.0f}")
        print(f"   MAE:  ¥{metrics['MAE']:,.0f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   R²:   {metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'RMSE', 'MAE', 'MAPE', 'R2']]

    return results_df


def plot_comparison(
    predictions: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    save_path: str = "figures/"
) -> None:
    """比較結果をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    # === 1. 予測結果の時系列比較 ===
    fig, ax = plt.subplots(figsize=(14, 6))

    # 実績（どちらのDataFrameでも同じはず）
    first_df = list(predictions.values())[0]
    ax.plot(first_df['date'], first_df['actual'],
            label='実績', linewidth=2, color='black', marker='o', markersize=3)

    # 各モデルの予測
    colors = {'Prophet': '#ff6b6b', 'LightGBM': '#4dabf7'}
    for model_name, df in predictions.items():
        ax.plot(df['date'], df['prediction'],
                label=f'{model_name}予測', linewidth=2,
                linestyle='--', color=colors.get(model_name, 'gray'))

    ax.set_title('実績 vs 各モデル予測', fontsize=14, fontweight='bold')
    ax.set_xlabel('日付')
    ax.set_ylabel('売上（円）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}10_model_comparison_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 2. 評価指標の比較（棒グラフ） ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    colors = ['#ff6b6b', '#4dabf7']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = metrics_df[metric].values
        models = metrics_df['model'].values

        bars = ax.bar(models, values, color=colors)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')

        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            if metric in ['RMSE', 'MAE']:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'¥{val:,.0f}', ha='center', va='bottom', fontsize=10)
            elif metric == 'MAPE':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        # R2は高いほど良い、他は低いほど良い
        if metric == 'R2':
            ax.set_ylabel('スコア（高いほど良い）')
        else:
            ax.set_ylabel('誤差（低いほど良い）')

    plt.tight_layout()
    plt.savefig(f"{save_path}11_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 3. 残差分析 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_name, df) in enumerate(predictions.items()):
        ax = axes[idx]
        residuals = df['actual'] - df['prediction']

        ax.scatter(df['prediction'], residuals, alpha=0.6,
                   color=colors[idx], s=30)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{model_name} 残差プロット', fontsize=14, fontweight='bold')
        ax.set_xlabel('予測値')
        ax.set_ylabel('残差（実績 - 予測）')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}12_residual_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 4. 誤差の分布 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_name, df) in enumerate(predictions.items()):
        ax = axes[idx]
        errors = df['actual'] - df['prediction']

        ax.hist(errors, bins=30, edgecolor='white', alpha=0.7, color=colors[idx])
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=2,
                   label=f'平均誤差: ¥{errors.mean():,.0f}')
        ax.set_title(f'{model_name} 誤差の分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('誤差（実績 - 予測）')
        ax.set_ylabel('頻度')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}13_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 比較グラフを保存しました")


def analyze_by_segment(predictions: Dict[str, pd.DataFrame]) -> None:
    """セグメント別の分析"""
    print("\n" + "=" * 60)
    print("📊 セグメント別分析")
    print("=" * 60)

    # 曜日別の誤差
    for model_name, df in predictions.items():
        df['day_of_week'] = df['date'].dt.dayofweek
        df['error'] = df['actual'] - df['prediction']
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = np.abs(df['error'] / df['actual']) * 100

        print(f"\n【{model_name}】曜日別 平均絶対誤差率:")
        dow_names = ['月', '火', '水', '木', '金', '土', '日']
        dow_errors = df.groupby('day_of_week')['pct_error'].mean()

        for dow, error in dow_errors.items():
            print(f"   {dow_names[dow]}曜日: {error:.2f}%")


def winner_summary(metrics_df: pd.DataFrame) -> None:
    """勝者を発表"""
    print("\n" + "=" * 60)
    print("🏆 総合評価")
    print("=" * 60)

    # 各指標での勝者
    print("\n各指標でのベストモデル:")

    # RMSE, MAE, MAPEは低いほど良い
    for metric in ['RMSE', 'MAE', 'MAPE']:
        best_idx = metrics_df[metric].idxmin()
        best_model = metrics_df.loc[best_idx, 'model']
        best_value = metrics_df.loc[best_idx, metric]
        if metric in ['RMSE', 'MAE']:
            print(f"   {metric}: {best_model} (¥{best_value:,.0f})")
        else:
            print(f"   {metric}: {best_model} ({best_value:.2f}%)")

    # R2は高いほど良い
    best_idx = metrics_df['R2'].idxmax()
    best_model = metrics_df.loc[best_idx, 'model']
    best_value = metrics_df.loc[best_idx, 'R2']
    print(f"   R²: {best_model} ({best_value:.4f})")

    # 総合勝者（R2で判定）
    overall_best = metrics_df.loc[metrics_df['R2'].idxmax(), 'model']
    print(f"\n🥇 今回のデータでは【{overall_best}】が優勢！")
    print("\n※ただし、データやユースケースによって結果は変わります")
    print("   - 説明性が重要 → Prophet（成分分解が見やすい）")
    print("   - 精度が最優先 → LightGBM（特徴量エンジニアリング次第）")
    print("   - 長期予測 → Prophet（トレンド捕捉が得意）")
    print("   - 短期予測 → LightGBM（ラグ特徴量が効く）")


def main():
    """メイン処理"""
    print("=" * 60)
    print("🔬 モデル比較 & 精度評価")
    print("=" * 60)

    # 予測結果を読み込み
    predictions = load_predictions()

    # モデル比較
    metrics_df = compare_models(predictions)

    # 比較プロット
    plot_comparison(predictions, metrics_df)

    # セグメント別分析
    analyze_by_segment(predictions)

    # 総合評価
    winner_summary(metrics_df)

    # 結果を保存
    metrics_df.to_csv("model_comparison_results.csv", index=False)
    print("\n✅ 比較結果を model_comparison_results.csv に保存しました")

    return metrics_df


if __name__ == "__main__":
    metrics_df = main()
