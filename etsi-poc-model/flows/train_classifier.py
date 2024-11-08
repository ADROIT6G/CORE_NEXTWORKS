import os
import numpy as np
import pandas as pd
from prefect import flow, task, get_run_logger, tags
from prefect.blocks.system import Secret
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cloudpickle as cp
from prefect_aws import MinIOCredentials
from prefect_aws.s3 import S3Bucket


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
os.environ['AWS_ACCESS_KEY_ID'] = Secret.load('minio-user').get()
os.environ['AWS_SECRET_ACCESS_KEY'] = Secret.load('minio-pwd').get()
os.environ['MLFLOW_S3_ENDPOINT_URL'] = Secret.load('minio-data-url').get()
######################################################################################
@task
def fetch_data():

    minio_credentials = MinIOCredentials(
       minio_root_user = Secret.load("minio-user").get(),
       minio_root_password = Secret.load("minio-pwd").get()
    )
    s3_client = minio_credentials.get_boto3_session().client(
       service_name="s3",
       endpoint_url= Secret.load("minio-data-url").get()
    )
    bucket_name = "classifier"
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    dfs = []
    for file_key in csv_files:
        local_filename = file_key.split('/')[-1]  # Extract the filename
        s3_client.download_file(Bucket=bucket_name, Key=file_key, Filename=local_filename)
        df = pd.read_csv(local_filename)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    df = pd.DataFrame(combined_df, columns=['predicted_value_1','predicted_value_2','predicted_value_3','predicted_value_4','predicted_value_5', 'membership'])

    return combined_df
@task
def generate_training_data(combined_df):
        batch_size=64
        combined_df = combined_df.sample(frac=1).reset_index(drop=True) #shuffle rows
        X = combined_df[['predicted_value_1','predicted_value_2','predicted_value_3','predicted_value_4','predicted_value_5']].to_numpy()
        Y = combined_df['membership'].to_numpy()
        n_rows = (len(X) // batch_size) * batch_size

        train_x = np.asarray(X[:n_rows]).astype(np.float64)
        test_x = np.asarray(X[:64]).astype(np.float64)
        train_y = np.asarray(Y[:n_rows]).astype(np.intc)
        test_y = np.asarray(Y[:64]).astype(np.intc)
        return train_x,train_y, test_x, test_y
@task
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
@task
def log_model(train_x,train_y, test_x, test_y):
    mlflow.set_tracking_uri(Secret.load('mlflow-url').get())
    mlflow.set_experiment('Classifier')
    with mlflow.start_run():
    #if True:
            alpha = 1
            l1_ratio = 0.5
            lr = RandomForestClassifier()
            lr.fit(train_x, train_y)
            predicted_qualities = lr.predict(test_x)
            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            #
        # Infer the model signature
            y_pred = predicted_qualities.astype(np.intc)
            print(y_pred)
            signature = infer_signature(test_x, y_pred)
        # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="sklearn-model",
                signature=signature,
                registered_model_name="Classifier",
            )
    test_y = test_y.reshape(-1)
    print(test_y)
    cm = confusion_matrix(np.asarray(test_y).astype(np.intc), np.asarray(y_pred).astype(np.intc))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.show()

@flow
def MLbasic(userdata: str = "Train"): 
    df = fetch_data()
    train_x,train_y, test_x, test_y = generate_training_data(df)

    log_model(train_x,train_y, test_x, test_y)
