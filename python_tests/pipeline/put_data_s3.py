import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3'])

import boto3
from toolkit import download_data

# Download data
download_data()

# Put data in S3
s3 = boto3.resource('s3')
s3.Object('ais-argo-artifacts-test', 'bank-full.csv').put(Body=open('bank-full.csv', 'rb'))
