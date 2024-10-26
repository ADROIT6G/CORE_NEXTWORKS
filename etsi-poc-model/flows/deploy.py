environment = "seldon"

import yaml
import prefect
from prefect import task, flow, tags, get_run_logger, variables
from kubernetes import client, config
import mlflow
from mlflow import MlflowClient
from prefect.blocks.system import Secret

# image: seldonio/mlflowserver:1.18.2

seldon_deployment = """
apiVersion: machinelearning.seldon.io/v1alpha3
kind: SeldonDeployment
metadata:
  name: mlflow
spec:
  name: avm-battery-prediction
  predictors:
  - componentSpecs:
    - spec:
        protocol: v2
        containers:
        - name: classifier
          resources:
            requests:
              memory: "512Mi"  # Adjust this based on your application's needs
              cpu: "500m"      # Adjust this based on your application's needs
            limits:
              memory: "1Gi"    # Set a maximum memory limit
              cpu: "1"         # Set a maximum CPU limit
          livenessProbe:
            initialDelaySeconds: 500
            failureThreshold: 500
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/ping
              port: http
              scheme: HTTP
          readinessProbe:
            initialDelaySeconds: 500
            failureThreshold: 500
            periodSeconds: 5
            successThreshold: 1
            httpGet:
              path: /health/ping
              port: http
              scheme: HTTP
    graph:
      children: []
      implementation: MLFLOW_SERVER
      modelUri: s3://mlflow/1/16f54b71379947d99ec5683c2a886628/artifacts/avm-model
      envSecretRefName: bpk-seldon-init-container-secret
#      envSecretRefName: seldon-init-container-secret
      name: classifier
    name: demo
    replicas: 1
"""

CUSTOM_RESOURCE_INFO = dict(
    group="machinelearning.seldon.io",
    version="v1alpha3",
    plural="seldondeployments",
)


@task
def get_model_location(model_name, model_version):
    mlflow.set_tracking_uri(Secret.load('mlflow-url').get())
    client = MlflowClient()

    if model_version:
        model_metadata = client.get_model_version(model_name, model_version)
        # Here, model_metadata is a single ModelVersion object
        latest_model_location = model_metadata.source
    else:
        model_metadata_list = client.get_latest_versions(model_name)
        # Here, model_metadata_list is a list of ModelVersion objects
        latest_model_location = model_metadata_list[0].source if model_metadata_list else None

    return latest_model_location


@task
def deploy_model(model_uri: str, serving_name: str, namespace: str = "seldon"):
    logger = get_run_logger()

    logger.info(f"Deploying model {model_uri} to enviroment {namespace}")

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    dep = yaml.safe_load(seldon_deployment)
    dep["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri
    dep["metadata"]["name"] = serving_name

    try:
        resp = custom_api.create_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            body=dep,
        )

        logger.info("Deployment created. status='%s'" % resp["status"]["state"])
    except:
        logger.info("Updating existing model")
        existing_deployment = custom_api.get_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=dep["metadata"]["name"],
        )
        existing_deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

        resp = custom_api.replace_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=existing_deployment["metadata"]["name"],
            body=existing_deployment,
        )

@flow    
def deploy(model_name: str, serving_name:str, model_version = None):
        
    model_uri = get_model_location(model_name, model_version)

    deploy_model(model_uri, serving_name, namespace="seldon")
