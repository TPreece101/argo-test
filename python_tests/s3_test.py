import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'boto3'])

import boto3
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('ais-argo-artifacts-test')

for my_bucket_object in my_bucket.objects.all():
    print(my_bucket_object)