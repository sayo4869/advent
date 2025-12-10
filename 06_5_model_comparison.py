"""
Prophet vs LightGBM vs Chronos vs Chronos-Bolt 全モデル比較
〜10種類の時系列予測手法を徹底比較〜

比較するモデル:
- Prophet: Metaの時系列予測ライブラリ（分解可能、解釈性高い）
- LightGBM: 勾配ブースティング（特徴量エンジニアリング重要）
- Chronos-tiny:  8M params（軽量、高速）
- Chronos-small: 46M params（バランス良い）
- Chronos-base:  200M params（高精度）
- Chronos-large: 710M params（最高精度）
- Bolt-tiny:  9M params（超高速）
- Bolt-mini:  21M params（高速）
- Bolt-small: 48M params（バランス良い）
- Bolt-base:  205M params（高精度）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import warnings
import japanize_matplotlib
import os

warnings.filterwarnings('ignore')

# Chronosモデルサイズ
CHRONOS_SIZES = ["tiny", "small", "base", "large"]

# Chronos-Boltモデルサイズ
BOLT_SIZES = ["tiny", "mini", "small", "base"]

# モデルの色設定
MODEL_COLORS = {
    'Prophet': '#ff6b6b',
    'LightGBM': '#4dabf7',
    'Chronos-tiny': '#a8e6cf',
    'Chronos-small': '#feca57',
    'Chronos-base': '#48dbfb',
    'Chronos-large': '#5f27cd',
    'Bolt-tiny': '#ff9ff3',
    'Bolt-mini': '#f368e0',
    'Bolt-small': '#ee5a24',
    'Bolt-base': '#c23616'
}


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_all_predictions() -> dict[str, pd.DataFrame]:
    """全モデルの予測結果を読み込む"""
    predictions = {}

    # Prophet
    if os.path.exists("prophet_predictions.csv"):
        print("📂 Prophet予測を読み込み中...")
        prophet_df = pd.read_csv("prophet_predictions.csv")
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.rename(columns={'ds': 'date', 'y': 'actual', 'yhat': 'prediction'})
        predictions['Prophet'] = prophet_df
        print(f"   ✅ {len(prophet_df)}件")
    else:
        print("   ⚠️ prophet_predictions.csv が見つかりません。スキップします。")

    # LightGBM
    if os.path.exists("lightgbm_predictions.csv"):
        print("📂 LightGBM予測を読み込み中...")
        lgb_df = pd.read_csv("lightgbm_predictions.csv")
        lgb_df['date'] = pd.to_datetime(lgb_df['date'])
        lgb_df = lgb_df.rename(columns={'sales': 'actual'})
        predictions['LightGBM'] = lgb_df
        print(f"   ✅ {len(lgb_df)}件")
    else:
        print("   ⚠️ lightgbm_predictions.csv が見つかりません。スキップします。")

    # Chronos（全サイズ）
    for size in CHRONOS_SIZES:
        filename = f"chronos_predictions_{size}.csv"
        print(f"📂 Chronos-{size}予測を読み込み中...")
        if os.path.exists(filename):
            chronos_df = pd.read_csv(filename)
            chronos_df['date'] = pd.to_datetime(chronos_df['date'])
            chronos_df = chronos_df.rename(columns={'sales': 'actual'})
            predictions[f'Chronos-{size}'] = chronos_df
            print(f"   ✅ {len(chronos_df)}件")
        else:
            print(f"   ⚠️ {filename} が見つかりません。スキップします。")

    # Chronos-Bolt（全サイズ）
    for size in BOLT_SIZES:
        filename = f"chronos_bolt_predictions_{size}.csv"
        print(f"📂 Bolt-{size}予測を読み込み中...")
        if os.path.exists(filename):
            bolt_df = pd.read_csv(filename)
            bolt_df['date'] = pd.to_datetime(bolt_df['date'])
            bolt_df = bolt_df.rename(columns={'sales': 'actual'})
            predictions[f'Bolt-{size}'] = bolt_df
            print(f"   ✅ {len(bolt_df)}件")
        else:
            print(f"   ⚠️ {filename} が見つかりません。スキップします。")

    return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """評価指標を計算"""
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def compare_all_models(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全モデルの比較"""
    print("\n" + "=" * 60)
    print("📊 全モデル比較")
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


def plot_all_models_comparison(
    predictions: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    save_path: str = "figures/"
) -> None:
    """全モデルの比較結果をプロット"""
    os.makedirs(save_path, exist_ok=True)

    # === 1. 予測結果の時系列比較 ===
    fig, ax = plt.subplots(figsize=(16, 8))

    # 実績
    first_df = list(predictions.values())[0]
    ax.plot(first_df['date'], first_df['actual'],
            label='実績', linewidth=2.5, color='black', marker='o', markersize=2)

    # 各モデルの予測
    for model_name, df in predictions.items():
        ax.plot(df['date'], df['prediction'],
                label=f'{model_name}',
                linewidth=1.5, linestyle='--',
                color=MODEL_COLORS.get(model_name, 'gray'))

    ax.set_title('実績 vs 全モデル予測比較', fontsize=14, fontweight='bold')
    ax.set_xlabel('日付')
    ax.set_ylabel('売上（円）')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}15_all_models_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 2. 評価指標の比較（棒グラフ） ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_list = ['RMSE', 'MAE', 'MAPE', 'R2']
    bar_colors = [MODEL_COLORS.get(m, 'gray') for m in metrics_df['model']]

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 2, idx % 2]
        values = metrics_df[metric].values
        models = metrics_df['model'].values

        bars = ax.bar(range(len(models)), values, color=bar_colors)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=60, ha='right', fontsize=8)

        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            if metric in ['RMSE', 'MAE']:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'¥{val:,.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
            elif metric == 'MAPE':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=7, rotation=45)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        if metric == 'R2':
            ax.set_ylabel('スコア（高いほど良い）')
        else:
            ax.set_ylabel('誤差（低いほど良い）')

    plt.tight_layout()
    plt.savefig(f"{save_path}16_all_models_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 3. R²スコアランキング ===
    fig, ax = plt.subplots(figsize=(12, 8))

    sorted_df = metrics_df.sort_values('R2', ascending=True)
    colors = [MODEL_COLORS.get(m, 'gray') for m in sorted_df['model']]

    bars = ax.barh(sorted_df['model'], sorted_df['R2'], color=colors)

    # 値を表示
    for bar, val in zip(bars, sorted_df['R2']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_xlabel('R²スコア（高いほど良い）')
    ax.set_title('モデル別 R²スコア ランキング', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f"{save_path}17_r2_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 4. 残差分析（上位3モデル） ===
    top3 = metrics_df.nlargest(3, 'R2')['model'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, model_name in enumerate(top3):
        ax = axes[idx]
        df = predictions[model_name]
        residuals = df['actual'] - df['prediction']

        ax.scatter(df['prediction'], residuals, alpha=0.6,
                   color=MODEL_COLORS.get(model_name, 'gray'), s=30)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{model_name} 残差プロット', fontsize=12, fontweight='bold')
        ax.set_xlabel('予測値')
        ax.set_ylabel('残差（実績 - 予測）')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}18_top3_residuals.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 5. 誤差分布（上位3モデル） ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, model_name in enumerate(top3):
        ax = axes[idx]
        df = predictions[model_name]
        errors = df['actual'] - df['prediction']

        ax.hist(errors, bins=30, edgecolor='white', alpha=0.7,
                color=MODEL_COLORS.get(model_name, 'gray'))
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=2,
                   label=f'平均誤差: ¥{errors.mean():,.0f}')
        ax.set_title(f'{model_name} 誤差の分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('誤差（実績 - 予測）')
        ax.set_ylabel('頻度')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}19_top3_error_dist.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === 6. モデルタイプ別比較 ===
    fig, ax = plt.subplots(figsize=(12, 6))

    # カテゴリ別にベストモデルを選択
    type_models = []
    type_labels = []

    if 'Prophet' in predictions:
        type_models.append('Prophet')
        type_labels.append('Prophet')

    if 'LightGBM' in predictions:
        type_models.append('LightGBM')
        type_labels.append('LightGBM')

    # Chronosベスト
    chronos_models = [m for m in metrics_df['model'] if m.startswith('Chronos-')]
    if chronos_models:
        best_chronos = metrics_df[metrics_df['model'].isin(chronos_models)].nlargest(1, 'R2')['model'].iloc[0]
        type_models.append(best_chronos)
        type_labels.append(f'Chronos\n({best_chronos.split("-")[1]})')

    # Boltベスト
    bolt_models = [m for m in metrics_df['model'] if m.startswith('Bolt-')]
    if bolt_models:
        best_bolt = metrics_df[metrics_df['model'].isin(bolt_models)].nlargest(1, 'R2')['model'].iloc[0]
        type_models.append(best_bolt)
        type_labels.append(f'Bolt\n({best_bolt.split("-")[1]})')

    if type_models:
        type_df = metrics_df[metrics_df['model'].isin(type_models)]
        x = np.arange(len(type_models))

        colors = [MODEL_COLORS.get(m, 'gray') for m in type_models]
        r2_values = [type_df[type_df['model'] == m]['R2'].values[0] for m in type_models]

        bars = ax.bar(x, r2_values, color=colors)

        for bar, val in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('R²スコア')
        ax.set_title('モデルタイプ別 ベストモデル比較', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_path}20_model_type_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    # === 7. Chronos vs Bolt 比較 ===
    if chronos_models and bolt_models:
        fig, ax = plt.subplots(figsize=(14, 6))

        chronos_df = metrics_df[metrics_df['model'].isin(chronos_models)].sort_values('model')
        bolt_df = metrics_df[metrics_df['model'].isin(bolt_models)].sort_values('model')

        x = np.arange(max(len(chronos_df), len(bolt_df)))
        width = 0.35

        # Chronos
        chronos_r2 = chronos_df['R2'].values
        ax.bar(x[:len(chronos_r2)] - width/2, chronos_r2, width,
               label='Chronos', color='#3498db')

        # Bolt
        bolt_r2 = bolt_df['R2'].values
        ax.bar(x[:len(bolt_r2)] + width/2, bolt_r2, width,
               label='Bolt', color='#e74c3c')

        ax.set_ylabel('R²スコア')
        ax.set_title('Chronos vs Chronos-Bolt サイズ別比較', fontsize=14, fontweight='bold')
        ax.set_xticks(x[:max(len(chronos_df), len(bolt_df))])
        labels = ['tiny', 'small/mini', 'base', 'large'][:max(len(chronos_df), len(bolt_df))]
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_path}21_chronos_vs_bolt.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ 比較グラフを保存しました（15〜21）")


def winner_summary(metrics_df: pd.DataFrame) -> None:
    """勝者を発表"""
    print("\n" + "=" * 60)
    print("🏆 全モデル総合評価")
    print("=" * 60)

    # 評価結果テーブル
    print("\n📊 評価結果一覧:")
    print(metrics_df.to_string(index=False))

    # 各指標での勝者
    print("\n各指標でのベストモデル:")

    for metric in ['RMSE', 'MAE', 'MAPE']:
        best_idx = metrics_df[metric].idxmin()
        best_model = metrics_df.loc[best_idx, 'model']
        best_value = metrics_df.loc[best_idx, metric]
        if metric in ['RMSE', 'MAE']:
            print(f"   {metric}: {best_model} (¥{best_value:,.0f})")
        else:
            print(f"   {metric}: {best_model} ({best_value:.2f}%)")

    best_idx = metrics_df['R2'].idxmax()
    best_model = metrics_df.loc[best_idx, 'model']
    best_value = metrics_df.loc[best_idx, 'R2']
    print(f"   R²: {best_model} ({best_value:.4f})")

    # ランキング
    print("\n📊 R²スコア ランキング:")
    ranking = metrics_df.sort_values('R2', ascending=False)
    medals = ['🥇', '🥈', '🥉'] + [f'{i}.' for i in range(4, 20)]
    for i, row in enumerate(ranking.itertuples()):
        medal = medals[i] if i < len(medals) else f'{i+1}.'
        print(f"   {medal} {row.model}: R²={row.R2:.4f}, MAPE={row.MAPE:.2f}%")

    # 総合勝者
    overall_best = metrics_df.loc[metrics_df['R2'].idxmax(), 'model']
    print(f"\n🎉 総合1位: 【{overall_best}】")

    # カテゴリ別ベスト
    print("\n" + "-" * 40)
    print("📝 カテゴリ別ベスト:")
    print("-" * 40)

    # 従来手法ベスト
    traditional = metrics_df[metrics_df['model'].isin(['Prophet', 'LightGBM'])]
    if len(traditional) > 0:
        best_trad = traditional.loc[traditional['R2'].idxmax(), 'model']
        print(f"   従来手法ベスト: {best_trad}")

    # Chronosベスト
    chronos = metrics_df[metrics_df['model'].str.startswith('Chronos-')]
    if len(chronos) > 0:
        best_chronos = chronos.loc[chronos['R2'].idxmax(), 'model']
        print(f"   Chronosベスト: {best_chronos}")

    # Boltベスト
    bolt = metrics_df[metrics_df['model'].str.startswith('Bolt-')]
    if len(bolt) > 0:
        best_bolt = bolt.loc[bolt['R2'].idxmax(), 'model']
        print(f"   Boltベスト: {best_bolt}")

    # 各モデルの特徴
    print("\n" + "-" * 40)
    print("📝 各モデルの特徴:")
    print("-" * 40)
    print("""
【Prophet】
  - 強み: 成分分解、解釈性、イベント効果
  - 向いている場面: ビジネスレポート、長期トレンド

【LightGBM】
  - 強み: 高精度、多変量対応、高速
  - 向いている場面: 短期予測、特徴量が豊富なデータ

【Chronos（従来版）】
  - tiny/small/base/large の4サイズ
  - 強み: ゼロショット、高精度
  - 弱み: 推論速度がやや遅い

【Chronos-Bolt（高速版）】
  - tiny/mini/small/base の4サイズ
  - 強み: 従来比最大250倍高速、分位点直接出力
  - 向いている場面: リアルタイム予測、大量バッチ処理
""")


def main():
    """メイン処理"""
    print("=" * 60)
    print("🔬 全モデル比較")
    print("   Prophet / LightGBM / Chronos / Chronos-Bolt")
    print("=" * 60)

    # 予測結果を読み込み
    predictions = load_all_predictions()

    if len(predictions) == 0:
        print("❌ 予測結果が見つかりません。各モデルを先に実行してください。")
        return None

    # モデル比較
    metrics_df = compare_all_models(predictions)

    # 比較プロット
    plot_all_models_comparison(predictions, metrics_df)

    # 総合評価
    winner_summary(metrics_df)

    # 結果を保存
    metrics_df.to_csv("all_models_comparison.csv", index=False)
    print("\n✅ 比較結果を all_models_comparison.csv に保存しました")

    return metrics_df


if __name__ == "__main__":
    metrics_df = main()
