"""
アパレル会社の売上ダミーデータ生成スクリプト
〜リアルな季節性とイベント効果を盛り込んだ2年弱分のデータ〜
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 再現性のためシードを固定（推しの番号でもOK）
np.random.seed(42)

def generate_apparel_sales_data(
    start_date: str = "2022-01-01",
    end_date: str = "2023-10-31",
    base_sales: float = 1000000
) -> pd.DataFrame:
    """
    アパレル会社の日次売上データを生成

    Parameters
    ----------
    start_date : str
        開始日
    end_date : str
        終了日
    base_sales : float
        基準売上（円）

    Returns
    -------
    pd.DataFrame
        日次売上データ
    """

    # 日付範囲を生成
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # === 1. トレンド成分 ===
    # 緩やかな成長トレンド（年率5%成長くらい）
    trend = np.linspace(0, 0.1, n_days)

    # === 2. 年間季節性 ===
    # アパレルは春夏と秋冬で大きく変わる
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])

    # 春物（3-4月）、夏物セール（7-8月）、秋冬物（10-11月）、冬セール（1月）がピーク
    seasonal_yearly = (
        0.15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)  # 春のピーク
        + 0.20 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # 夏セール
        + 0.10 * np.sin(2 * np.pi * (day_of_year - 300) / 365)  # 秋冬
    )

    # === 3. 週次季節性 ===
    # 土日は売上UP、月曜は閑散
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_pattern = {
        0: -0.15,  # 月曜：みんなお疲れ
        1: -0.08,  # 火曜
        2: -0.05,  # 水曜
        3: 0.00,   # 木曜
        4: 0.10,   # 金曜：週末前のお買い物
        5: 0.25,   # 土曜：かき入れ時！
        6: 0.20,   # 日曜：午後から減速
    }
    seasonal_weekly = np.array([weekly_pattern[dow] for dow in day_of_week])

    # === 4. イベント効果 ===
    events = []
    event_effects = np.zeros(n_days)

    for i, date in enumerate(dates):
        month, day = date.month, date.day

        # 初売り（1/1-1/3）：爆売れ
        if month == 1 and day <= 3:
            event_effects[i] = 0.8
            events.append({"date": date, "event": "初売り"})

        # バレンタイン（2/10-14）
        elif month == 2 and 10 <= day <= 14:
            event_effects[i] = 0.2
            events.append({"date": date, "event": "バレンタイン"})

        # ホワイトデー（3/10-14）
        elif month == 3 and 10 <= day <= 14:
            event_effects[i] = 0.15
            events.append({"date": date, "event": "ホワイトデー"})

        # GW（4/29-5/5）
        elif (month == 4 and day >= 29) or (month == 5 and day <= 5):
            event_effects[i] = 0.35
            events.append({"date": date, "event": "GW"})

        # 夏のボーナスセール（6/25-7/10）
        elif (month == 6 and day >= 25) or (month == 7 and day <= 10):
            event_effects[i] = 0.45
            events.append({"date": date, "event": "夏ボーナスセール"})

        # お盆（8/10-16）
        elif month == 8 and 10 <= day <= 16:
            event_effects[i] = 0.25
            events.append({"date": date, "event": "お盆"})

        # シルバーウィーク（9/15-23あたり）
        elif month == 9 and 15 <= day <= 23:
            event_effects[i] = 0.2
            events.append({"date": date, "event": "シルバーウィーク"})

        # ハロウィン（10/25-31）
        elif month == 10 and day >= 25:
            event_effects[i] = 0.15
            events.append({"date": date, "event": "ハロウィン"})

        # ブラックフライデー（11/20-26あたり）
        elif month == 11 and 20 <= day <= 26:
            event_effects[i] = 0.5
            events.append({"date": date, "event": "ブラックフライデー"})

        # 冬のボーナスセール（12/1-15）
        elif month == 12 and day <= 15:
            event_effects[i] = 0.4
            events.append({"date": date, "event": "冬ボーナスセール"})

        # クリスマス（12/20-25）
        elif month == 12 and 20 <= day <= 25:
            event_effects[i] = 0.55
            events.append({"date": date, "event": "クリスマス"})

        # 年末（12/26-31）
        elif month == 12 and day >= 26:
            event_effects[i] = 0.3
            events.append({"date": date, "event": "年末"})

    # === 5. ノイズ ===
    # 現実世界は予測不能なこともある
    noise = np.random.normal(0, 0.08, n_days)

    # === 6. 売上を合成 ===
    multiplier = 1 + trend + seasonal_yearly + seasonal_weekly + event_effects + noise
    sales = base_sales * multiplier

    # 負の売上は0に（念のため）
    sales = np.maximum(sales, 0)

    # === データフレーム作成 ===
    df = pd.DataFrame({
        "date": dates,
        "sales": sales.astype(int),
        "day_of_week": day_of_week,
        "month": [d.month for d in dates],
        "day": [d.day for d in dates],
        "is_weekend": [1 if dow >= 5 else 0 for dow in day_of_week],
    })

    # イベントフラグを追加
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.groupby("date")["event"].first().reset_index()
        df = df.merge(events_df, on="date", how="left")
        df["event"] = df["event"].fillna("通常営業")
    else:
        df["event"] = "通常営業"

    return df


def main():
    """メイン処理"""
    print("=" * 50)
    print("🧥 アパレル売上ダミーデータ生成中...")
    print("=" * 50)

    # データ生成
    df = generate_apparel_sales_data()

    # 基本情報を表示
    print(f"\n📅 データ期間: {df['date'].min()} 〜 {df['date'].max()}")
    print(f"📊 レコード数: {len(df):,} 件")
    print(f"💰 売上統計:")
    print(f"   - 平均: ¥{df['sales'].mean():,.0f}")
    print(f"   - 最小: ¥{df['sales'].min():,.0f}")
    print(f"   - 最大: ¥{df['sales'].max():,.0f}")
    print(f"   - 標準偏差: ¥{df['sales'].std():,.0f}")

    # イベント別の売上
    print(f"\n🎉 イベント別平均売上:")
    event_sales = df.groupby("event")["sales"].mean().sort_values(ascending=False)
    for event, sales in event_sales.items():
        print(f"   - {event}: ¥{sales:,.0f}")

    # CSVに保存
    output_path = "apparel_sales_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ データを {output_path} に保存しました！")

    return df


if __name__ == "__main__":
    df = main()
