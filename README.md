# 小売売上予測シリーズ - Prophet vs LightGBM vs Chronos

Qiita アドベントカレンダー用の記事素材です。

小売業界の売上データを使って、時系列予測モデルを比較・実装・本番運用します。

## 記事構成

| 記事 | 内容 | ファイル |
|-----|------|---------|
| **第一弾** | Prophet vs LightGBM 基礎編 | [article.md](./article.md) |
| **第二弾** | Chronos（Transformer）体験編 | [article_part2.md](./article_part2.md) |
| **第三弾** | LightGBM + AWS本番運用編 | [article_part3.md](./article_part3.md) |

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

### 第一弾: Prophet vs LightGBM

```bash
python 01_generate_dummy_data.py   # ダミーデータ生成
python 02_preprocessing_eda.py     # 前処理 & EDA
python 03_prophet_forecast.py      # Prophet予測
python 04_lightgbm_forecast.py     # LightGBM予測
python 05_model_comparison.py      # モデル比較
```

### 第二弾: Chronos（Transformer）

```bash
python 06_chronos_forecast.py      # Chronos予測
python 07_batch_forecast.py        # バッチ予測（ローカル）
```

### 第三弾: AWS本番運用

```bash
# モデルをS3にアップロード
python 10_save_model_to_s3.py --bucket my-bucket

# Dockerビルド & デプロイ
cd aws
./deploy.sh
```

## ファイル構成

```
advent/
├── README.md
├── requirements.txt              # 開発用（全ライブラリ）
├── requirements-aws.txt          # 本番用（軽量）
│
├── article.md                    # 第一弾記事
├── article_part2.md              # 第二弾記事
├── article_part3.md              # 第三弾記事
│
├── 01_generate_dummy_data.py     # ダミーデータ生成
├── 02_preprocessing_eda.py       # 前処理・EDA
├── 03_prophet_forecast.py        # Prophet予測
├── 04_lightgbm_forecast.py       # LightGBM予測
├── 05_model_comparison.py        # モデル比較
├── 06_chronos_forecast.py        # Chronos予測
├── 07_batch_forecast.py          # バッチ予測（ローカル）
├── 08_batch_forecast_aws.py      # Chronos AWS版
├── 09_lightgbm_batch_aws.py      # LightGBM AWS版
├── 10_save_model_to_s3.py        # モデルS3保存
│
├── Dockerfile                    # 本番用コンテナ
├── aws/
│   ├── ecs-task-definition.json
│   ├── stepfunctions-definition.json
│   └── deploy.sh
│
├── figures/                      # (生成) 可視化グラフ
└── forecasts/                    # (生成) バッチ予測結果
```

## 主な内容

### 第一弾: 基礎編
- ダミーデータ生成（季節性・イベント効果）
- EDA（時系列プロット、季節性分析、ADF検定）
- Prophet vs LightGBM の比較

### 第二弾: Transformer編
- Chronos（Amazon開発）の紹介
- 特徴量エンジニアリング不要の新アプローチ
- 3モデル比較

### 第三弾: AWS運用編
- Dockerコンテナ化
- ECS Fargateでサーバーレス実行
- Step Functionsでワークフロー管理
- 月額$1以下で運用可能

## 注意点

- 時系列データでは**データリーク**に注意
- `shift(1)` で当日のデータを使わないようにする
- train/test分割は時間順（シャッフルNG）
- Chronosは**GPU推奨**（CPU可だが遅い）
- 本番運用は**LightGBM推奨**（軽量・高速）

## ライセンス

MIT
