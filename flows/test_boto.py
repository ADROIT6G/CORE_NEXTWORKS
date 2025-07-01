from prefect_aws import MinIOCredentials
from prefect_aws.s3 import S3Bucket
import pandas as pd
import numpy as np

def fetch_data():

    minio_credentials = MinIOCredentials(
       minio_root_user = "minio",
       minio_root_password = "minio123"
    )
    s3_client = minio_credentials.get_boto3_session().client(
       service_name="s3",
       endpoint_url="http://10.30.8.228:9001"
    )
    s3_bucket = S3Bucket(
        bucket_name="test1",  # must exist
        minio_credentials=MinIOCredentials(minio_root_user = "minio", minio_root_password = "minio123"),
        endpoint_url="http://10.30.8.228:9001"
    )

    s3_resource = minio_credentials.get_boto3_session().resource(
       service_name="s3",
       endpoint_url="http://10.30.8.228:9001"
    )
    s3_object = s3_resource.Object(
       bucket_name='test1',
       key='data.csv'
    )

#    s3_client.download_file('test1', 'data.csv', 'data.csv')
#    s3_client.download_file(Bucket="test1", Key="data.csv", Filename="data.csv")
    s3_object.download_file(Filename='./data.csv')
    #s3_bucket.download_object_to_path("data.csv", "data.csv")
    data = pd.read_csv("data.csv")
    print(data.head())

fetch_data()
