# SageMaker MLflow App (cloudformation/mlflow-app) and local MLflow helpers.
#
#   make deploy         create/update the mlflow-app stack in us-east-1
#   make outputs        print stack outputs (App ARN, MLflow version, bucket)
#   make url            open a presigned URL to the MLflow App UI
#   make teardown       delete the stack (the artifact bucket is retained)
#   make mlflow-ui      browse the local SQLite store used by the scripts
#   make image          build the BYOC scorer image (sagemaker-ai/container/) for local docker run
#   make image-push     create the ECR repo if needed, then push the image

-include .env
export

AWS_PROFILE ?= aws-free-tier
AWS_REGION ?= us-east-1
STACK ?= mlflow-app
STACK_DIR := cloudformation/mlflow-app
IMAGE_REPO ?= compileml-scorer
ACCOUNT_ID = $(shell $(AWS) sts get-caller-identity --query Account --output text)
IMAGE_URI = $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(IMAGE_REPO):latest
AWS := aws --profile $(AWS_PROFILE) --region $(AWS_REGION)

.PHONY: deploy validate outputs arn url teardown mlflow-ui image image-push

validate:
	cd $(STACK_DIR) && sam validate

deploy: validate
	cd $(STACK_DIR) && sam deploy --no-progressbar

outputs:
	$(AWS) cloudformation describe-stacks --stack-name $(STACK) \
	  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table

arn:
	@$(AWS) cloudformation describe-stacks --stack-name $(STACK) \
	  --query 'Stacks[0].Outputs[?OutputKey==`MlflowAppArn`].OutputValue' --output text

url:
	@$(AWS) sagemaker create-presigned-mlflow-app-url --arn $$($(MAKE) -s arn) \
	  --expires-in-seconds 300 --query AuthorizedUrl --output text | xargs open

teardown:
	cd $(STACK_DIR) && sam delete --no-prompts

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Local-only build for the host architecture (quick `docker run compileml-scorer:local serve`).
image:
	docker build -t $(IMAGE_REPO):local sagemaker-ai/container

# Multi-arch push: SageMaker endpoints run linux/amd64; ModelBuilder's local container mode
# pulls whatever matches the host (arm64 on Apple Silicon), and a single-arch manifest list
# fails that pull with "no matching manifest for linux/arm64/v8".
image-push:
	$(AWS) ecr describe-repositories --repository-names $(IMAGE_REPO) >/dev/null 2>&1 || \
	  $(AWS) ecr create-repository --repository-name $(IMAGE_REPO) --image-scanning-configuration scanOnPush=true >/dev/null
	$(AWS) ecr get-login-password | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(IMAGE_URI) sagemaker-ai/container
