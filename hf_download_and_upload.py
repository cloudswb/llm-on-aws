import argparse
import boto3
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download


def download_llm_from_huggingface(model_name, token, cache_path, max_call_times):
    local_cache_path = Path(cache_path)
    local_cache_path.mkdir(parents=True, exist_ok=True)
    is_download_successed = False
    allow_patterns = ["*.json", "*.pt", "*.bin", "*.model"]
    model_download_path = ""

    def download_recursively(max_calls):
        nonlocal is_download_successed
        if max_calls <= 0 or is_download_successed:
            return model_download_path

        try:
            model_download_path = snapshot_download(
                repo_id=model_name,
                cache_dir=local_cache_path,
                allow_patterns=allow_patterns,
                local_files_only=False,
                token=token,
            )
            if model_download_path:
                is_download_successed = True
                print("SUCCESS DOWNLOADED:", model_download_path)
                return model_download_path
            else:
                return download_recursively(max_calls - 1)
        except Exception as e:
            print("Download failed:", e)
            return download_recursively(max_calls - 1)

    return download_recursively(max_call_times)
    # Return the downloaded path

def upload_to_s3(aws_access_key, aws_secret_key, aws_region, s3_model_loc):
    print(f"Starting upload from {model_download_path} to {s3_model_loc}")
    # Set AWS credentials
    subprocess.run(["aws", "configure", "set", "aws_access_key_id", aws_access_key])
    subprocess.run(["aws", "configure", "set", "aws_secret_access_key", aws_secret_key])
    subprocess.run(["aws", "configure", "set", "default.region", aws_region])

    # Upload to S3
    s3_upload_command = ["aws", "s3", "cp", f"{model_download_path}", s3_model_loc, "--recursive"]
    upload_result = subprocess.run(s3_upload_command)
    
    print(f"upload_result.returncode:{upload_result.returncode}")
    if upload_result.returncode != 0:
        raise Exception("Upload to S3 failed. Exiting...")
    else:
        print(f"Upload to s3 successed:{s3_model_loc}")

def create_s3_bucket_if_not_exists(aws_access_key, aws_secret_key, aws_region, s3_model_loc):
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
    parts = s3_model_loc.split('/')
    bucket_name = parts[2]

    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except NoCredentialsError:
        print("AWS credentials are not provided or invalid.")
        raise
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            # The bucket does not exist; create it
            print("The bucket does not exist; create it.")
            s3_client.create_bucket(Bucket=bucket_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and upload files to Amazon S3")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--token", required=True, help="Model token information")
    parser.add_argument("--cache_path", required=True, help="Download cache path")
    parser.add_argument("--max_call_times", type=int, default=3, help="Maximum download retry times")
    parser.add_argument("--aws_access_key", required=True, help="AWS access key ID")
    parser.add_argument("--aws_secret_key", required=True, help="AWS secret access key")
    parser.add_argument("--aws_region", required=True, help="AWS region")
    parser.add_argument("--s3_model_loc", required=True, help="S3 destination path")

    args = parser.parse_args()

    try:
        model_download_path = download_llm_from_huggingface(args.model_name, args.token, args.cache_path, args.max_call_times)
        assert(model_download_path)
        create_s3_bucket_if_not_exists(args.aws_access_key, args.aws_secret_key, args.aws_region, args.s3_model_loc)
        upload_to_s3(args.aws_access_key, args.aws_secret_key, args.aws_region, args.s3_model_loc)
    except Exception as e:
        print("Error:", e)

