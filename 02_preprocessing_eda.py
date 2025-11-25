"""
前処理 & EDA（探索的データ分析）
〜データと仲良くなる時間〜
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Tuple
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    データの読み込みと前処理

    Parameters
    ----------
    filepath : str
        CSVファイルのパス

    Returns
    -------
    pd.DataFrame
        前処理済みデータ
    """
    print("📂 データを読み込み中...")
    df = pd.read_csv(filepath)

    # 日付型に変換
    df['date'] = pd.to_datetime(df['date'])

    # === 基本的なデータ確認 ===
    print("\n" + "=" * 50)
    print("📊 データの基本情報")
    print("=" * 50)
    print(f"\n形状: {df.shape}")
    print(f"\nカラム: {list(df.columns)}")
    print(f"\nデータ型:\n{df.dtypes}")

    # 欠損値チェック
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ 欠損値あり:\n{missing[missing > 0]}")
    else:
        print("\n✅ 欠損値なし！")

    # 重複チェック
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ 重複行: {duplicates} 件")
        df = df.drop_duplicates()
        print("   → 重複を削除しました")
    else:
        print("✅ 重複なし！")

    # === 追加の特徴量エンジニアリング ===
    print("\n🔧 特徴量を追加中...")

    # 年、四半期
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter

    # 週番号
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # 月初・月末フラグ
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 給料日周辺（25日前後はお財布が温かい）
    df['is_payday_period'] = ((df['day'] >= 24) & (df['day'] <= 28)).astype(int)

    # イベントフラグ（通常営業以外）
    df['has_event'] = (df['event'] != '通常営業').astype(int)

    print(f"   追加した特徴量: year, quarter, week_of_year, is_month_start, is_month_end, is_payday_period, has_event")

    return df


def basic_statistics(df: pd.DataFrame) -> None:
    """基本統計量の表示"""
    print("\n" + "=" * 50)
    print("📈 売上の基本統計量")
    print("=" * 50)

    stats = df['sales'].describe()
    print(f"\n{stats}")

    # パーセンタイル
    print(f"\n📊 パーセンタイル:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = df['sales'].quantile(p / 100)
        print(f"   {p}%: ¥{val:,.0f}")


def plot_time_series(df: pd.DataFrame, save_path: str = "figures/") -> None:
    """時系列プロット"""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. 日次売上の推移
    ax1 = axes[0]
    ax1.plot(df['date'], df['sales'], linewidth=0.8, alpha=0.7)
    ax1.set_title('日次売上の推移', fontsize=14, fontweight='bold')
    ax1.set_ylabel('売上（円）')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.grid(True, alpha=0.3)

    # イベント期間をハイライト
    event_df = df[df['has_event'] == 1]
    ax1.scatter(event_df['date'], event_df['sales'], c='red', s=10, alpha=0.5, label='イベント日')
    ax1.legend()

    # 2. 月次売上（集計）
    ax2 = axes[1]
    monthly = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()
    monthly.index = monthly.index.to_timestamp()
    ax2.bar(monthly.index, monthly.values, width=25, alpha=0.7, color='steelblue')
    ax2.set_title('月次売上合計', fontsize=14, fontweight='bold')
    ax2.set_ylabel('売上（円）')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 7日移動平均
    ax3 = axes[2]
    df['sales_ma7'] = df['sales'].rolling(window=7, center=True).mean()
    df['sales_ma30'] = df['sales'].rolling(window=30, center=True).mean()
    ax3.plot(df['date'], df['sales'], alpha=0.3, linewidth=0.5, label='日次')
    ax3.plot(df['date'], df['sales_ma7'], linewidth=1.5, label='7日移動平均')
    ax3.plot(df['date'], df['sales_ma30'], linewidth=2, label='30日移動平均')
    ax3.set_title('移動平均による平滑化', fontsize=14, fontweight='bold')
    ax3.set_ylabel('売上（円）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}01_time_series.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}01_time_series.png を保存しました")


def plot_seasonality(df: pd.DataFrame, save_path: str = "figures/") -> None:
    """季節性の分析"""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 曜日別売上
    ax1 = axes[0, 0]
    dow_names = ['月', '火', '水', '木', '金', '土', '日']
    dow_sales = df.groupby('day_of_week')['sales'].mean()
    colors = ['#ff6b6b' if i >= 5 else '#4dabf7' for i in range(7)]
    ax1.bar(dow_names, dow_sales.values, color=colors)
    ax1.set_title('曜日別平均売上', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均売上（円）')
    ax1.axhline(y=df['sales'].mean(), color='red', linestyle='--', label='全体平均')
    ax1.legend()

    # 2. 月別売上
    ax2 = axes[0, 1]
    month_sales = df.groupby('month')['sales'].mean()
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
    ax2.bar(range(1, 13), month_sales.values, color=colors)
    ax2.set_title('月別平均売上', fontsize=14, fontweight='bold')
    ax2.set_xlabel('月')
    ax2.set_ylabel('平均売上（円）')
    ax2.set_xticks(range(1, 13))

    # 3. 曜日×月のヒートマップ
    ax3 = axes[1, 0]
    pivot = df.pivot_table(values='sales', index='day_of_week', columns='month', aggfunc='mean')
    pivot.index = dow_names
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '売上'})
    ax3.set_title('曜日×月 平均売上ヒートマップ', fontsize=14, fontweight='bold')
    ax3.set_xlabel('月')
    ax3.set_ylabel('曜日')

    # 4. イベント別売上
    ax4 = axes[1, 1]
    event_sales = df.groupby('event')['sales'].mean().sort_values(ascending=True)
    colors = ['#69db7c' if e == '通常営業' else '#ffd43b' for e in event_sales.index]
    ax4.barh(event_sales.index, event_sales.values, color=colors)
    ax4.set_title('イベント別平均売上', fontsize=14, fontweight='bold')
    ax4.set_xlabel('平均売上（円）')
    ax4.axvline(x=df['sales'].mean(), color='red', linestyle='--', label='全体平均')

    plt.tight_layout()
    plt.savefig(f"{save_path}02_seasonality.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}02_seasonality.png を保存しました")


def plot_distribution(df: pd.DataFrame, save_path: str = "figures/") -> None:
    """分布の分析"""
    import os
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ヒストグラム
    ax1 = axes[0, 0]
    ax1.hist(df['sales'], bins=50, edgecolor='white', alpha=0.7, color='steelblue')
    ax1.axvline(df['sales'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均: ¥{df["sales"].mean():,.0f}')
    ax1.axvline(df['sales'].median(), color='orange', linestyle='--', linewidth=2, label=f'中央値: ¥{df["sales"].median():,.0f}')
    ax1.set_title('売上の分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('売上（円）')
    ax1.set_ylabel('頻度')
    ax1.legend()

    # 2. 箱ひげ図（曜日別）
    ax2 = axes[0, 1]
    dow_names = ['月', '火', '水', '木', '金', '土', '日']
    df['dow_name'] = df['day_of_week'].map(dict(enumerate(dow_names)))
    df.boxplot(column='sales', by='dow_name', ax=ax2,
               positions=[0, 1, 2, 3, 4, 5, 6])
    ax2.set_title('曜日別売上の箱ひげ図', fontsize=14, fontweight='bold')
    ax2.set_xlabel('曜日')
    ax2.set_ylabel('売上（円）')
    plt.suptitle('')  # 自動タイトルを削除

    # 3. QQプロット（正規性の確認）
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(df['sales'], dist="norm", plot=ax3)
    ax3.set_title('QQプロット（正規性の確認）', fontsize=14, fontweight='bold')

    # 4. 対数変換後のヒストグラム
    ax4 = axes[1, 1]
    log_sales = np.log1p(df['sales'])
    ax4.hist(log_sales, bins=50, edgecolor='white', alpha=0.7, color='coral')
    ax4.set_title('売上の分布（対数変換後）', fontsize=14, fontweight='bold')
    ax4.set_xlabel('log(売上+1)')
    ax4.set_ylabel('頻度')

    plt.tight_layout()
    plt.savefig(f"{save_path}03_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}03_distribution.png を保存しました")


def check_stationarity(df: pd.DataFrame) -> None:
    """定常性の確認（ADF検定）"""
    from statsmodels.tsa.stattools import adfuller

    print("\n" + "=" * 50)
    print("📉 定常性の確認（ADF検定）")
    print("=" * 50)

    result = adfuller(df['sales'].dropna())
    print(f"\nADF統計量: {result[0]:.4f}")
    print(f"p値: {result[1]:.4f}")
    print(f"使用したラグ数: {result[2]}")

    if result[1] < 0.05:
        print("\n✅ p < 0.05: データは定常であると判断できます")
    else:
        print("\n⚠️ p >= 0.05: データは非定常の可能性があります")
        print("   → 差分を取るか、トレンド除去を検討しましょう")


def correlation_analysis(df: pd.DataFrame, save_path: str = "figures/") -> None:
    """相関分析"""
    import os
    os.makedirs(save_path, exist_ok=True)

    # 自己相関を計算
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ACF（自己相関関数）
    plot_acf(df['sales'].dropna(), lags=40, ax=axes[0])
    axes[0].set_title('自己相関関数（ACF）', fontsize=14, fontweight='bold')

    # PACF（偏自己相関関数）
    plot_pacf(df['sales'].dropna(), lags=40, ax=axes[1], method='ywm')
    axes[1].set_title('偏自己相関関数（PACF）', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_path}04_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {save_path}04_correlation.png を保存しました")


def summary_report(df: pd.DataFrame) -> None:
    """EDAサマリーレポート"""
    print("\n" + "=" * 60)
    print("📋 EDA サマリーレポート")
    print("=" * 60)

    print(f"""
【データ概要】
・期間: {df['date'].min().strftime('%Y-%m-%d')} 〜 {df['date'].max().strftime('%Y-%m-%d')}
・レコード数: {len(df):,} 件

【売上統計】
・平均売上: ¥{df['sales'].mean():,.0f}
・中央値: ¥{df['sales'].median():,.0f}
・標準偏差: ¥{df['sales'].std():,.0f}
・変動係数: {df['sales'].std() / df['sales'].mean():.2%}

【季節性の特徴】
・最も売れる曜日: {['月','火','水','木','金','土','日'][df.groupby('day_of_week')['sales'].mean().idxmax()]}曜日
・最も売れる月: {df.groupby('month')['sales'].mean().idxmax()}月
・週末効果: +{((df[df['is_weekend']==1]['sales'].mean() / df[df['is_weekend']==0]['sales'].mean()) - 1) * 100:.1f}%

【イベント効果】
・イベント日の売上増加率: +{((df[df['has_event']==1]['sales'].mean() / df[df['has_event']==0]['sales'].mean()) - 1) * 100:.1f}%
・最も売れるイベント: {df.groupby('event')['sales'].mean().idxmax()}
""")


def main():
    """メイン処理"""
    # データ読み込みと前処理
    df = load_and_preprocess("apparel_sales_data.csv")

    # 基本統計量
    basic_statistics(df)

    # 可視化
    print("\n📊 グラフを作成中...")
    plot_time_series(df)
    plot_seasonality(df)
    plot_distribution(df)
    correlation_analysis(df)

    # 定常性チェック
    check_stationarity(df)

    # サマリーレポート
    summary_report(df)

    # 前処理済みデータを保存
    df.to_csv("apparel_sales_preprocessed.csv", index=False, encoding="utf-8")
    print("\n✅ 前処理済みデータを apparel_sales_preprocessed.csv に保存しました")

    return df


if __name__ == "__main__":
    df = main()
