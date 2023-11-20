# Databricks notebook source
# MAGIC %md
# MAGIC ###### Initialization
# MAGIC
# MAGIC We will initialize few variables and declare some utility functions
# MAGIC
# MAGIC We can always invoke this notebook from other notebooks using the `%run` magic command. Variables and methods defined in this notebook will be available in the calling notebook. 
# MAGIC
# MAGIC To know more read the documentation [here](https://docs.databricks.com/en/notebooks/notebook-workflows.html)

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
user_name = user_email.split('@')[0].replace('.','_')
user_prefix = f"{user_name[0:4]}{str(len(user_name)).rjust(3, '0')}"

# COMMAND ----------

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root_path = '/'.join(current_path.split("/")[:-1])

# COMMAND ----------

## CHANGE raw_data_path TO THE LOCATION WHERE RAW DATA FILE NEED TO BE COPIED
#raw_data_path = "/Volumes/main/diabetes_prediction/raw_data"
raw_data_path = "s3://databricks-e2demofieldengwest/external_location_srijit_nair"


# COMMAND ----------

import requests
db_host = spark.conf.get('spark.databricks.workspaceUrl')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
db_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
headers = {"Authorization": f"Bearer {db_token}"}
url = f"https://{db_host}/api/2.0/clusters/get?cluster_id={db_cluster_id}"
response = requests.get(url, headers=headers).json()
uc_enabled = False if response["data_security_mode"]=="NONE" else True

# COMMAND ----------

catalog = "main"
database = "diabetes_prediction"

demographic_table = f"{database}.patient_demographics_{user_prefix}"
lab_results_table = f"{database}.patient_lab_results_{user_prefix}"
physicals_results_table = f"{database}.patient_pysicals_{user_prefix}"
feature_table_name = f"{database}.diabetes_features_{user_prefix}"
inference_data_table_nonfs = f"{database}.patient_data_nonfs_{user_prefix}"
inference_data_table_fs = f"{database}.patient_data_fs_{user_prefix}"
raw_data_table = f"{database}.diabetes_raw_{user_prefix}"
volume_raw_data = "uc_not_available"
model_registry_uri = "databricks"
registered_model_name_non_fs = f"{user_prefix}_diabetes_prediction_nonfs"
registered_model_name_fs = f"{user_prefix}_diabetes_prediction_fs"

print(f"***************************************************")
print(f"Unity Catalog is { 'Enabled' if uc_enabled else 'Not Enabled' }")
print(" ")

if uc_enabled :
  demographic_table = f"{catalog}.{demographic_table}"
  lab_results_table = f"{catalog}.{lab_results_table}"
  physicals_results_table = f"{catalog}.{physicals_results_table}"
  feature_table_name =  f"{catalog}.{feature_table_name}"
  inference_data_table_nonfs = f"{catalog}.{inference_data_table_nonfs}"
  inference_data_table_fs = f"{catalog}.{inference_data_table_fs}"
  raw_data_table = f"{catalog}.{raw_data_table}"

  model_registry_uri = "databricks-uc"
  registered_model_name_non_fs = f"{catalog}.{database}.{registered_model_name_non_fs}"
  registered_model_name_fs = f"{catalog}.{database}.{registered_model_name_fs}"  

if raw_data_path.startswith("/Volumes/"):
  volume_raw_data = raw_data_path.replace("/Volumes/","").replace("/",".")

print(f"Project Root Path(project_root_path):{project_root_path}")
print(" ")
print(f"Demographic table (demographic_table): {demographic_table}")
print(f"Lab Result table (lab_results_table): {lab_results_table}")
print(f"Physicals Result table (physicals_results_table): {physicals_results_table}")
print(f"Feature table (feature_table_name): {feature_table_name}")
print(f"Inference Data table Non Feature Store (inference_data_table_nonfs): {inference_data_table_nonfs}")
print(f"Inference Data table Feature Store (inference_data_table_fs): {inference_data_table_fs}")
print(f"Raw Data Table for EDA(raw_data_table): {raw_data_table}")
print(f"Raw Data path(raw_data_path): {raw_data_path}")
print(f"UC Volume for raw data(volume_raw_data): {volume_raw_data}")
print(" ")
print(f"Model Registry URI (model_registry_uri): {model_registry_uri}")
print(f"Model Name Non Feature Store (registered_model_name_non_fs): {registered_model_name_non_fs}")
print(f"Model Name With Feature Store (registered_model_name_fs): {registered_model_name_fs}")
print(f"***************************************************")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

# COMMAND ----------


#Gets the latest model details
def get_latest_model_version(model_name: str, env_or_alias: str=""):  
  #If UC is not enabled we will use workspace registry
  if not uc_enabled:
    mlflow.set_registry_uri("databricks")
    client = MlflowClient()
    models = client.get_latest_versions(model_name, stages=[env_or_alias])
    if len(models) >0:
      return models[0]
    else:
      return None
  #If UC is enabled we will use UC api
  else:
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    if env_or_alias == "":
      models = client.search_model_versions(f"name='{model_name}'")
      if len(models) >0:
        return models[0]
      else:
        return None
    else:
      try:
        return client.get_model_version_by_alias(name=model_name,alias=env_or_alias)
      except:
        return None


# COMMAND ----------

def get_model_uri(model_info,env_or_alias):
  if model_info:
    if uc_enabled:
      return f"models:/{model_info.name}@{env_or_alias}"
    else:
      return f"models:/{model_info.name}/{env_or_alias}"
  else:
    raise Exception("No model versions are registered for production use")

# COMMAND ----------


