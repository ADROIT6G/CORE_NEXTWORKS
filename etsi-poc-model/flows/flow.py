from prefect import flow, task, get_run_logger, tags
from prefect.blocks.system import Secret
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cloudpickle as cp
import anfis_layers 
from prefect_aws import MinIOCredentials
from prefect_aws.s3 import S3Bucket

os.environ['AWS_ACCESS_KEY_ID'] = Secret.load('minio-user').get()
os.environ['AWS_SECRET_ACCESS_KEY'] = Secret.load('minio-pwd').get()
os.environ['MLFLOW_S3_ENDPOINT_URL'] = Secret.load('mlflow-url').get()

param = anfis_layers.fis_parameters(
        n_input=4,                # no. of Regressors
        n_memb=3,                 # no. of fuzzy memberships
        batch_size=64,            # 16 / 32 / 64 / ...
        memb_func='gaussian',     # 'gaussian' / 'gbellmf' / 'sigmoid' /'triangle'(use rmsprop)
        optimizer='adam',      # sgd / adam / rmsprop /...
        loss='mse',               # mse / mae / huber_loss / mean_absolute_percentage_error / ...
        n_epochs=34,              # 10 / 25 / 50 / 100 / ...
        mf_range = (-2,2),            # range of membership functions ((0.7,1.3),
        memberships = ['URLLC_BytesReceived', 'URLLC_BytesSent', 'URLLC_Received_thrp_Mbps', 'URLLC_Sent_thrp_Mbps']
    )

@task
def save_pickle(filename, object_s):
        with open(filename, 'wb') as f:
            cp.dump(object_s, f)
######################################################################################
#data
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
    bucket_name = "trainingdata"
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    dfs = []
    for file_key in csv_files:
        local_filename = file_key.split('/')[-1]  # Extract the filename
        s3_client.download_file(Bucket=bucket_name, Key=file_key, Filename=local_filename)
        df = pd.read_csv(local_filename)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

@task
def generate_training_data(combined_df):
        combined_df = combined_df.sample(frac=1).reset_index(drop=True) #shuffle rows
        X = combined_df[['URLLC_BytesReceived', 'URLLC_BytesSent', 'URLLC_Received_thrp_Mbps', 'URLLC_Sent_thrp_Mbps']].to_numpy()
        Y = combined_df['membership'].to_numpy()
        n_rows = (len(X) // param.batch_size) * param.batch_size
        return X[:n_rows], Y[:n_rows], combined_df
@task
def normalize_data(array, columns_to_normalize):
        scaler = MinMaxScaler(feature_range=(0, 1))
        array[:, columns_to_normalize] = scaler.fit_transform(array[:, columns_to_normalize])
        save_pickle('scaler_training_data.pkl', scaler)
        return array, scaler
######################################################################################
#model
@task
def ANFISModel(parameters):
        anfis = anfis_layers.ANFIS(
            n_input=parameters.n_input,
            n_memb=parameters.n_memb,
            batch_size=parameters.batch_size,
            memb_func=parameters.memb_func,
            mf_range = parameters.mf_range,
            name='ANFIS',
            memberships = parameters.memberships
        )
        return anfis
@task
def compile_model(anfis, parameters):
        anfis.model.compile(optimizer=parameters.optimizer, loss=parameters.loss, metrics=['mae', 'mse'])
@task
def fit_model(anfis, parameters, X_train, y_train, X_test, y_test):
        return anfis.fit(X_train, y_train, epochs=parameters.n_epochs, batch_size=parameters.batch_size, validation_data=(X_test, y_test))
@task
def set_weights(anfis, base_model):
        return anfis.model.set_weights(base_model.model.get_weights())
@task
def log_model(parameters, model, model_name, example_data, experiment):
        mlflow.set_tracking_uri(os.environ['MLFLOW_S3_ENDPOINT_URL'])
        mlflow.set_experiment(experiment)
        print(f"this is the MLFlow tracking URI: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
        with mlflow.start_run():
            mlflow.log_params(parameters.__dict__)
            mlflow.log_artifact("memberships.png")
            mlflow.log_artifact("memb_curves.npy")
            data = np.random.rand(1, parameters.n_input)
            answer = model(data)
            signature = infer_signature(data, answer)
            mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path=model_name,
                signature = signature,
                code_paths=["etsi-poc-model/flows/anfis_layers.py", "scaler_training_data.pkl"],
                input_example=example_data,
            )
@task
def model_save(anfis, modelname):
        anfis.model.save(modelname)
@task
def plot_membership_functions(anfis):
        membership_plot = anfis.plotmfs( show_initial_weights=True)
        anfis.model.summary()
        return membership_plot

######################################################################################

@flow
def MLbasic(userdata: str = "Train"):       
# Data Processing
        data = fetch_data()

        logger = get_run_logger()
        logger.info(f"minIO data: {data.head(10)}!")

        X_raw, Y_switch, combined_df = generate_training_data(data)
        X, scaler = normalize_data(X_raw, [0, 1, 2, 3])

        # Model Training
        anfis = ANFISModel(param)
        compile_model(anfis, param)
        fit_model(anfis, param, X, Y_switch, X, Y_switch) 
        plot_membership_functions(anfis)

    # change batch size to 1 for inference
        param.batch_size = 1
        input_data = X[:64]
        model_to_save = ANFISModel(param)
        compile_model(anfis, param)
        set_weights(model_to_save, anfis)
        mlflow_url = Secret.load("mlflow-url").get()
                #debug
        # Path to the file
        file_path = "memberships.png"

        # Check if the file exists
        if os.path.exists(file_path):
                print(f"{file_path} was found successfully.")
        else:
                print(f"{file_path} does not exist.")
        log_model(param, model_to_save, "ANFIS", input_data, 'ANFIS')
