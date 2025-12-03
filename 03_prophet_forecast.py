"""
Prophet による小売売上の時系列予測
〜Meta社のライブラリで未来を占う〜
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（平均絶対パーセント誤差）を計算"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # ゼロ除算を避ける
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet用にデータを整形

    Prophetはデータフレームにdsとyのカラムが必要
    ds: 日付
    y: 予測対象
    """
    prophet_df = df[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df


def create_japanese_holidays() -> pd.DataFrame:
    """
    日本の小売に関連するイベントを定義

    Prophetはholidayとして外部イベントを考慮できる
    """
    # 2022年〜2025年のイベント
    holidays = []

    for year in [2022, 2023, 2024, 2025]:
        # 初売り
        for day in range(1, 4):
            holidays.append({
                'holiday': '初売り',
                'ds': f'{year}-01-0{day}',
                'lower_window': 0,
                'upper_window': 0,
            })

        # バレンタイン
        for day in range(10, 15):
            holidays.append({
                'holiday': 'バレンタイン',
                'ds': f'{year}-02-{day}',
                'lower_window': 0,
                'upper_window': 0,
            })

        # GW
        holidays.extend([
            {'holiday': 'GW', 'ds': f'{year}-04-29', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-04-30', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-05-01', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-05-02', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-05-03', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-05-04', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'GW', 'ds': f'{year}-05-05', 'lower_window': 0, 'upper_window': 0},
        ])

        # 夏のボーナスセール（6月は30日まで）
        for day in range(25, 31):
            holidays.append({
                'holiday': '夏ボーナスセール',
                'ds': f'{year}-06-{day}',
                'lower_window': 0,
                'upper_window': 0,
            })
        for day in range(1, 11):
            holidays.append({
                'holiday': '夏ボーナスセール',
                'ds': f'{year}-07-{day:02d}',
                'lower_window': 0,
                'upper_window': 0,
            })

        # お盆
        for day in range(10, 17):
            holidays.append({
                'holiday': 'お盆',
                'ds': f'{year}-08-{day}',
                'lower_window': 0,
                'upper_window': 0,
            })

        # ブラックフライデー
        for day in range(20, 27):
            holidays.append({
                'holiday': 'ブラックフライデー',
                'ds': f'{year}-11-{day}',
                'lower_window': 0,
                'upper_window': 0,
            })

        # クリスマス
        for day in range(20, 26):
            holidays.append({
                'holiday': 'クリスマス',
                'ds': f'{year}-12-{day}',
                'lower_window': 0,
                'upper_window': 0,
            })

    holidays_df = pd.DataFrame(holidays)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return holidays_df


def train_prophet_model(
    train_df: pd.DataFrame,
    holidays: pd.DataFrame = None,
    add_country_holidays: bool = True
) -> Prophet:
    """
    Prophetモデルを学習

    Parameters
    ----------
    train_df : pd.DataFrame
        学習データ（ds, y列を持つ）
    holidays : pd.DataFrame
        カスタム休日データ
    add_country_holidays : bool
        日本の祝日を追加するか

    Returns
    -------
    Prophet
        学習済みモデル
    """
    print("🔮 Prophetモデルを構築中...")

    # モデルの初期化
    model = Prophet(
        growth='linear',                    # 成長モデル（linear or logistic）
        seasonality_mode='multiplicative',  # 季節性モード（additive or multiplicative）
        yearly_seasonality=True,            # 年間季節性
        weekly_seasonality=True,            # 週次季節性
        daily_seasonality=False,            # 日次季節性（日次データなので不要）
        holidays=holidays,
        changepoint_prior_scale=0.05,       # トレンド変化点の柔軟性
        seasonality_prior_scale=10,         # 季節性の強さ
        holidays_prior_scale=10,            # 休日効果の強さ
        interval_width=0.95,                # 予測区間の幅
    )

    # 日本の祝日を追加
    if add_country_holidays:
        model.add_country_holidays(country_name='JP')

    # カスタム季節性を追加（月次パターン）
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )

    # 学習
    print("   学習中...")
    model.fit(train_df)
    print("   ✅ 学習完了！")

    return model


def evaluate_prophet(
    model: Prophet,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Prophetモデルを評価

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        予測結果のDataFrameと評価指標の辞書
    """
    print("\n📊 モデルを評価中...")

    # テストデータで予測
    future = test_df[['ds']].copy()
    forecast = model.predict(future)

    # 実績と予測を結合
    results = test_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds'
    )

    # 評価指標を計算
    y_true = results['y'].values
    y_pred = results['yhat'].values

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

    print("\n📈 Prophet 評価結果:")
    print(f"   RMSE: ¥{metrics['RMSE']:,.0f}")
    print(f"   MAE:  ¥{metrics['MAE']:,.0f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")

    return results, metrics


def plot_prophet_results(
    model: Prophet,
    forecast: pd.DataFrame,
    train_df: pd.DataFrame,
    test_results: pd.DataFrame,
    save_path: str = "figures/"
) -> None:
    """結果をプロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    # 1. Prophetの標準プロット
    fig1 = model.plot(forecast)
    plt.title('Prophet 予測結果', fontsize=14, fontweight='bold')
    plt.xlabel('日付')
    plt.ylabel('売上（円）')
    plt.tight_layout()
    plt.savefig(f"{save_path}05_prophet_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. コンポーネント分解
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(f"{save_path}06_prophet_components.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. テスト期間の実績 vs 予測
    fig3, ax = plt.subplots(figsize=(14, 6))

    ax.plot(test_results['ds'], test_results['y'],
            label='実績', linewidth=2, marker='o', markersize=3)
    ax.plot(test_results['ds'], test_results['yhat'],
            label='予測', linewidth=2, linestyle='--')
    ax.fill_between(
        test_results['ds'],
        test_results['yhat_lower'],
        test_results['yhat_upper'],
        alpha=0.3, label='95%信頼区間'
    )

    ax.set_title('テスト期間: 実績 vs 予測 (Prophet)', fontsize=14, fontweight='bold')
    ax.set_xlabel('日付')
    ax.set_ylabel('売上（円）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}07_prophet_test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Prophetのグラフを保存しました")


def cross_validation_prophet(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    """
    時系列クロスバリデーション

    注意: ProphetのCVは計算に時間がかかる
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    print("\n🔄 クロスバリデーション中...")

    # CVを実行
    # initial: 初期学習期間
    # period: 各カットオフ間の間隔
    # horizon: 予測期間
    df_cv = cross_validation(
        model,
        initial='365 days',
        period='30 days',
        horizon='30 days',
        parallel="processes"
    )

    # パフォーマンス指標を計算
    df_perf = performance_metrics(df_cv)

    print("\n📊 クロスバリデーション結果:")
    print(df_perf[['horizon', 'mse', 'rmse', 'mae', 'mape']].tail())

    return df_cv, df_perf


def main():
    """メイン処理"""
    print("=" * 60)
    print("🔮 Prophet による小売売上予測")
    print("=" * 60)

    # データ読み込み
    df = pd.read_csv("retail_sales_preprocessed.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Prophet用に整形
    prophet_df = prepare_data_for_prophet(df)

    # 学習・テストデータに分割
    # 最後の2ヶ月をテストデータに
    split_date = prophet_df['ds'].max() - pd.Timedelta(days=60)
    train_df = prophet_df[prophet_df['ds'] <= split_date].copy()
    test_df = prophet_df[prophet_df['ds'] > split_date].copy()

    print(f"\n📅 データ分割:")
    print(f"   学習データ: {train_df['ds'].min().strftime('%Y-%m-%d')} 〜 {train_df['ds'].max().strftime('%Y-%m-%d')} ({len(train_df)}件)")
    print(f"   テストデータ: {test_df['ds'].min().strftime('%Y-%m-%d')} 〜 {test_df['ds'].max().strftime('%Y-%m-%d')} ({len(test_df)}件)")

    # カスタム休日を作成
    holidays = create_japanese_holidays()

    # モデル学習
    model = train_prophet_model(train_df, holidays=holidays)

    # 全期間の予測（プロット用）
    future_all = prophet_df[['ds']].copy()
    forecast_all = model.predict(future_all)

    # テストデータで評価
    test_results, metrics = evaluate_prophet(model, test_df)

    # 結果をプロット
    plot_prophet_results(model, forecast_all, train_df, test_results)

    # 結果を保存
    test_results.to_csv("prophet_predictions.csv", index=False)
    print("\n✅ 予測結果を prophet_predictions.csv に保存しました")

    # メトリクスを保存（後で比較用）
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'Prophet'
    metrics_df.to_csv("prophet_metrics.csv", index=False)

    return model, test_results, metrics


if __name__ == "__main__":
    model, results, metrics = main()
