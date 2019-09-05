import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3'])

import os
import boto3
from toolkit import download_data

# Get S3 bucket from environment
s3_bucket = os.getenv('S3BUCKET')

# Download data
download_data()

# Put data in S3
s3 = boto3.resource('s3')
s3.Object(s3_bucket, 'bank-full.csv').put(Body=open('bank-full.csv', 'rb'))
