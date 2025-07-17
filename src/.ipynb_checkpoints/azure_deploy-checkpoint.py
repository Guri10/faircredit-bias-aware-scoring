from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint, Model

from azure.identity import DefaultAzureCredential

# Fill these in:
subscription_id = "1fe7c5d7-59c1-44be-9eae-932fa4606402"
resource_group  = "faircredit-rg"
workspace_name  = "faircredit-ws"
# model_uri       = "models:/faircredit_baseline/bdd9f70f60f747f1a159a16e8468d5c0
model_uri = "mlruns/875467197912374825/models/m-8edaed32179b46bc9292c9dd5469f2c5/artifacts"


ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)


# run_id = "bdd9f70f60f747f1a159a16e8468d5c0"  
# model_uri = f"runs:/{run_id}/model"

# Register the MLflow run artifact as an Azure ML model asset
model = ml_client.models.create_or_update(
    Model(
        name="faircredit-baseline",
        path=model_uri,
        type="mlflow_model"
    )
)
print(f"Registered model from {model_uri}")


# 2) Define environment from your Conda spec
env = Environment(
    name="faircredit-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest",
)
ml_client.environments.create_or_update(env)

# 3) Create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="faircredit-endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 4) Deploy
deployment = ManagedOnlineDeployment(
    name="prod-deployment",
    endpoint_name=endpoint.name,
    model=model,
    environment=env,
    code_configuration={"code": ".", "scoring_script": "score.py"},
    instance_type="Standard_DS2_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

print("Endpoint deployed at:", endpoint.scoring_uri)
