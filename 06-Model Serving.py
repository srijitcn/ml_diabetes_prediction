# Databricks notebook source
# MAGIC %md
# MAGIC #### Model Serving with Databricks
# MAGIC Model Serving exposes your MLflow machine learning models as scalable REST API endpoints and provides a highly available and low-latency service for deploying models. The service automatically scales up or down to meet demand changes, saving infrastructure costs while optimizing latency performance. This functionality uses serverless compute. 
# MAGIC
# MAGIC Model Serving offers:
# MAGIC - **Launch an endpoint with one click**: Databricks automatically prepares a production-ready environment for your model and offers serverless configuration options for compute.
# MAGIC - **High availability and scalability**: Model Serving is intended for production use and can support up to 25000+ queries-per-second (QPS). Model Serving endpoints automatically scale up and down, which means that endpoints automatically adjust based on the volume of scoring requests. You can also serve multiple models from a single endpoint.
# MAGIC - **Secure: Models are deployed in a secure network boundary**. Models use dedicated compute that terminates (and are never reused) when the model is deleted, or scaled down to zero.
# MAGIC - **MLflow integration**: Natively connects to the MLflow Model Registry which enables fast and easy deployment of models.
# MAGIC - **Quality and diagnostics**: Automatically capture requests and responses in a Delta table to monitor and debug models or generate training datasets. Endpoint health metrics, including QPS, latency, and error rates, are displayed in near-real time and can be exported to preferred observability tools.
# MAGIC - **Feature store integration**: When your model is trained with features from Databricks Feature Store, the model is packaged with feature metadata. If you configure your online store, these features are incorporated in real-time as scoring requests are received.
# MAGIC
# MAGIC ###### Additional resources
# MAGIC Detailed documentation for below features can be found [here](https://docs.databricks.com/en/machine-learning/model-serving/index.html#additional-resources)
# MAGIC - Model serving tutorial
# MAGIC - Optimized LLM serving
# MAGIC - Migrate to Model Serving
# MAGIC - Create and manage model serving endpoints
# MAGIC - Serving endpoints access control
# MAGIC - Send scoring requests to serving endpoints
# MAGIC - Serve multiple models to a Model Serving endpoint
# MAGIC - Use custom Python libraries with Model Serving
# MAGIC - Package custom artifacts for Model Serving
# MAGIC - Inference tables for monitoring and debugging models
# MAGIC - Enable inference tables on model serving endpoints
# MAGIC - Monitor served models with Lakehouse Monitoring
# MAGIC - Track and export serving endpoint health metrics to Prometheus and Datadog
# MAGIC - Deploy custom models with Model Serving
# MAGIC - Configure access to resources from model serving endpoints
# MAGIC - Add an instance profile to a model serving endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create endpoint from UI

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create endpoint via API

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Upgrade the databricks sdk

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.26.0 --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

env_or_alias = "champion" if uc_enabled else "production"
model_info = get_latest_model_version(registered_model_name_non_fs, env_or_alias)
model_uri = get_model_uri(model_info,env_or_alias)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import *

from datetime import timedelta
w = WorkspaceClient(host=db_host,token=db_token)

endpoint_name = "diabetes_pred_nonfs_endpoint"

served_entities = [ServedEntityInput(
                      name="diabetes_pred",
                      entity_name=registered_model_name_non_fs,
                      entity_version=model_info.version,
                      workload_size='Small',
                      workload_type='CPU', 
                      scale_to_zero_enabled=True)]


print(f"Creating model serving endpoint {endpoint_name}")

try:
  w.serving_endpoints.create_and_wait(name=endpoint_name, 
                                     config=EndpointCoreConfigInput(served_entities=served_entities), 
                                     timeout=timedelta(minutes=40)) 
except: 
  # when the endpoint already exists, update it

  w.serving_endpoints.update_config_and_wait(name=endpoint_name, 
                                             served_entities=served_entities, 
                                             timeout=timedelta(minutes=40))

# COMMAND ----------

endpoint_info = w.serving_endpoints.get(name=endpoint_name)

# COMMAND ----------

endpoint_info

# COMMAND ----------

# MAGIC %md
# MAGIC #### Score using model serve endpoint

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
 
def create_tf_serving_json(data):
  return {"inputs": {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}
 
def process_input(dataset):
  if isinstance(dataset, pd.DataFrame):
    return {"dataframe_split": dataset.to_dict(orient='split')}
  elif isinstance(dataset, str):
    return dataset
  else:
    return create_tf_serving_json(dataset)
 
def score_model(dataset):
  url = f"https://{db_host}/serving-endpoints/{endpoint_name}/invocations"
  print(f"Invoking {url}")
  headers = {"Authorization": f"Bearer {db_token}"}
  data_json = process_input(dataset)
  response = requests.request(method="POST", headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  return response.json()

# COMMAND ----------

data = pd.DataFrame(
  { 
      "Age" : [31,41,51],
      "BloodPressure" : [66,76,86],
      "Insulin" : [2,3,4],
      "BMI" : [24.5,25.5,26.5],
      "SkinThickness" : [29,30,31],
      "DiabetesPedigreeFunction" : [0.35, 0.45,0.55],
      "Pregnancies" : [0,1,2],
      "Glucose" :[85,95,105]
})

# COMMAND ----------

score_model(data)

# COMMAND ----------


