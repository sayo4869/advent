# アパレル売上予測 - Prophet vs LightGBM

Qiita アドベントカレンダー用の記事素材です。

アパレル業界の売上データを使って、Prophet と LightGBM による時系列予測を比較します。

## セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd advent

# 仮想環境の作成 & 有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# ライブラリのインストール
pip install -r requirements.txt
```

## 実行方法

順番に実行してください：

```bash
# 1. ダミーデータ生成
python 01_generate_dummy_data.py

# 2. 前処理 & EDA
python 02_preprocessing_eda.py

# 3. Prophet で予測
python 03_prophet_forecast.py

# 4. LightGBM で予測
python 04_lightgbm_forecast.py

# 5. モデル比較
python 05_model_comparison.py
```

## ファイル構成

```
advent/
├── README.md                     # このファイル
├── requirements.txt              # 必要なライブラリ
│
├── 01_generate_dummy_data.py     # ダミーデータ生成
├── 02_preprocessing_eda.py       # 前処理・EDA
├── 03_prophet_forecast.py        # Prophet予測
├── 04_lightgbm_forecast.py       # LightGBM予測
├── 05_model_comparison.py        # モデル比較
│
└── figures/                      # (生成) 可視化グラフ
    ├── ...
```

## 主な内容

1. **ダミーデータ生成**: アパレル業界特有の季節性（初売り、GW、ブラックフライデー等）を再現
2. **EDA**: 時系列プロット、季節性分析、定常性検定
3. **Prophet**: カスタム休日設定、成分分解
4. **LightGBM**: ラグ特徴量、ローリング特徴量（データリーク対策）
5. **精度評価**: RMSE, MAE, MAPE, R2 での比較

## 注意点

- 時系列データでは**データリーク**に注意
- `shift(1)` で当日のデータを使わないようにする
- train/test分割は時間順（シャッフルNG）

## ライセンス

MIT
