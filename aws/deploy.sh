#!/bin/bash
# デプロイスクリプト
# Usage: ./deploy.sh

set -e

# 設定（環境に応じて変更）
AWS_REGION="${AWS_REGION:-ap-northeast-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="sales-forecast"
S3_BUCKET="${S3_BUCKET:-my-forecast-bucket}"

echo "=========================================="
echo "🚀 Deploying Sales Forecast to AWS"
echo "=========================================="
echo "AWS Account: ${AWS_ACCOUNT_ID}"
echo "Region: ${AWS_REGION}"
echo "S3 Bucket: ${S3_BUCKET}"
echo ""

# 1. ECRリポジトリ作成（存在しない場合）
echo "📦 Creating ECR repository..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} 2>/dev/null || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME}

# 2. ECRにログイン
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 3. Dockerイメージをビルド
echo "🐳 Building Docker image..."
docker build -t ${ECR_REPO_NAME}:latest ..

# 4. タグ付け & プッシュ
echo "📤 Pushing to ECR..."
docker tag ${ECR_REPO_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest

# 5. S3バケット作成（存在しない場合）
echo "🪣 Creating S3 bucket..."
aws s3 mb s3://${S3_BUCKET} 2>/dev/null || echo "Bucket already exists"

# 6. CloudWatch Logsグループ作成
echo "📊 Creating CloudWatch Logs group..."
aws logs create-log-group --log-group-name /ecs/sales-forecast 2>/dev/null || echo "Log group already exists"

# 7. ECSタスク定義を登録
echo "📋 Registering ECS task definition..."
envsubst < ecs-task-definition.json > /tmp/task-def.json
aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json

echo ""
echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your model to S3:"
echo "   aws s3 cp lightgbm_model.pkl s3://${S3_BUCKET}/models/"
echo ""
echo "2. Upload your data to S3:"
echo "   aws s3 cp retail_sales_preprocessed.csv s3://${S3_BUCKET}/data/"
echo ""
echo "3. Create Step Functions state machine (optional)"
echo ""
