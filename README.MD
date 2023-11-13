# Deploy the model from HuggingFace to Sagemaker in China

## 1. Download mode from HuggingFace

Run the python script : hf_download_and_upload.py

### Parameter Description:

--model_name, the model name defined in HuggingFace, required

--token, the token used to download model from HuggingFace, optional

--cache_path, the cache files used in local path, required

--max_call_times, the download retry times, required

--aws_access_key, the AWS user access key in AWS account to upload to S3, required

--aws_secret_key, the AWS user secret key in AWS account to upload to S3, required

--aws_region, the Region code of AWS S3 located

--s3_model_loc, the bucket location used to upload

### Examples:

- Download the embedding model “BAAI/bge-large-en-v1.5”

```jsx
python3 hf_download_and_upload.py \
--model_name "BAAI/bge-large-en-v1.5" \
--token "YOUR_MODEL_TOKEN" \
--cache_path "../cache/model/BAAI/bge-large-en-v1.5" \
--max_call_times 10 \
--aws_access_key "AKIA3FXLNLQOIG73VBVJ" \
--aws_secret_key "jWY3Ym7HlYqX3oq/SZk3RB5YWwGoh/6Z34UhTnYg" \
--aws_region "cn-northwest-1" \
--s3_model_loc "s3://sagemaker-cn-northwest-1-768219110428/llm/model/BAAI/bge-large-en-v1.5/"
```

## 2. Deploy the model to Sagemaker

- Open/Create Sagemaker Notebook
- Download git source code: https://github.com/cloudswb/llm-on-aws.git
- Open the script (.ipynb file) from [deploy](https://github.com/cloudswb/llm-on-aws/tree/main/deploy) folder. Choose the right file depends on Model name.
- Review and config the parameters in **Step 2 section.**
    - Parameter Description:
        - model_name ： HuggingFace中的模型名称
        - s3_model_prefix ：模型文件在S3中的位置的文件夹路径（不包含bucket name和文件名称）, 【需要提前准备】
        - s3_code_prefix ： 模型执行代码在S3中的位置的文件夹路径（不包含bucket name和文件名称）
        - endpoint_config_name ： 部署Sagemaker Configuration 的名称
        - endpoint_name ： 部署Sagemaker endpoint的名称
        - deploy_cache_location ： 部署时，产生的代码文件所在的本地路径
        - inference_image_uri ： 部署所使用的推理容器
- Run the script to start deploy cell by cell or Run all cells
- Monitor the deploy progress
- Test the LLM