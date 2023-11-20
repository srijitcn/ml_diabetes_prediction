# Databricks notebook source
# MAGIC %md
# MAGIC #### Cleanup Data from S3 bucket
# MAGIC Please cleanup the S3 bucket manually

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delta Tables

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Feature Tables

# COMMAND ----------

#Remove the feature store entry
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
try:
  #Check if feature table exists. Delete if exists
  print(f"Deleting feature table {feature_table_name}")
  fs.drop_table(name=feature_table_name)
except:
  print(f"Feature table {feature_table_name} not found")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete Database and Delta Tables

# COMMAND ----------

#spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")

spark.sql(f"DROP TABLE IF EXISTS {demographic_table}")
spark.sql(f"DROP TABLE IF EXISTS {lab_results_table}")
spark.sql(f"DROP TABLE IF EXISTS {physicals_results_table}")
spark.sql(f"DROP TABLE IF EXISTS {feature_table_name}")
spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_nonfs}")
spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_fs}")
spark.sql(f"DROP TABLE IF EXISTS {raw_data_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove Image copied for Markdown

# COMMAND ----------

#Commenting this so that everyone can refer to these images later
#Uncomment if you want to delete the images

#tgt_folder = "/FileStore/tmp/diabetes_prediction"
#dbutils.fs.rm(tgt_folder, True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete Cache and Checkpoints

# COMMAND ----------

checkpoint_dir = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_checkpoint_weights"
path_to_cache = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_cache"
path_to_cache_dist = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_cache_dist"

dbutils.fs.rm(checkpoint_dir, True)
dbutils.fs.rm(path_to_cache, True)
dbutils.fs.rm(path_to_cache_dist, True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete endpoints

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import *

w = WorkspaceClient(host=db_host,token=db_token)

endpoint_name = f"{registered_model_name_non_fs}_endpoint"
try:
  print(f"Deleting endpoint {endpoint_name}")
  w.serving_endpoints.delete(name=endpoint_name)  
except:
  print(f"Endpoint {endpoint_name} does not exist")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete Registered Models

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri(model_registry_uri)
mlflow_client = MlflowClient()

# COMMAND ----------

if not uc_enabled:  
  models = mlflow_client.search_registered_models(filter_string=f"name ILIKE '%{user_prefix}_%'")
  for model in models:    
    model_versions = model.latest_versions
    for model_version in model_versions:
      if model_version.current_stage != "Archived":
        print(f"Archiving version {model_version.version} of model {model.name}")
        mlflow_client.transition_model_version_stage(name=model.name,version=model_version.version,stage="Archived")
    
    print(f"Deleting model {model.name}")
    mlflow_client.delete_registered_model(name=model.name)  

# COMMAND ----------

if uc_enabled:
  models = mlflow_client.search_registered_models()
  for model in models:    
    if model.name.startswith(f"{catalog}.{database}"):
      print(f"Deleting model {model.name}")
      mlflow_client.delete_registered_model(name=model.name) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delete Experiments

# COMMAND ----------


experiments = mlflow_client.search_experiments(filter_string=f"name ILIKE '%{user_prefix}_%'")
for experiment in experiments:
  print(f"Deleting experiment {experiment.name}")
  mlflow_client.delete_experiment(experiment_id=experiment.experiment_id)  

# COMMAND ----------

experiment_base_path = f"Users/{user_email}/mlflow_experiments"
dbutils.fs.rm(f"file:/Workspace/{experiment_base_path}",True)

# COMMAND ----------


