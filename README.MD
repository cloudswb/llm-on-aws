## Manually download model from huggingface and upload to S3 

### Preparation

- AWS Confidentional
- Model Name in HuggingFace
- HuggingFace token (depends on the model)
- Download parameters
  --model_name "BAAI/bge-large-en-v1.5" 
  --token "YOUR_MODEL_TOKEN" 
  --cache_path "./cache/BAAI/bge-large-en-v1.5" 
  --max_call_times 3 
  --aws_access_key "YOUR_AWS_ACCESS_KEY" 
  --aws_secret_key "YOUR_AWS_SECRET_KEY" 
  --aws_region "YOUR_AWS_REGION" 
  --s3_model_loc "s3://your-bucket/llm/rep/your-model-name"

python3 hf_download_and_upload.py --model_name "BAAI/bge-large-en-v1.5" --token "YOUR_MODEL_TOKEN" --cache_path "./cache/model/BAAI/bge-large-en-v1.5" --max_call_times 3 --aws_access_key "AKIA3FXLNLQOIG73VBVJ" --aws_secret_key "jWY3Ym7HlYqX3oq/SZk3RB5YWwGoh/6Z34UhTnYg" --aws_region "cn-northwest-1" --s3_model_loc "s3://sagemaker-cn-northwest-1-768219110428/llm/rep/BAAI/bge-large-en-v1.5/"