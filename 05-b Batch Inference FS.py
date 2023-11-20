# Databricks notebook source
# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Batch Inference Data
# MAGIC Now we will creatre some mock data to use for inference. 
# MAGIC
# MAGIC Since we are not using feature store, the dataframe is expected to have all the features required by the model

# COMMAND ----------

#We will do feature look up for ["Age", "BloodPressure", "BMI", "Pregnancies"]
feature_columns = ["Id","Insulin", "SkinThickness", "DiabetesPedigreeFunction", "Glucose"]

spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_fs}")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {inference_data_table_fs} AS
  SELECT {','.join(feature_columns)} FROM {feature_table_name} LIMIT 50
""")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {inference_data_table_fs}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC Let us lookup the model thats tagged as `Production` in model registry to use for our batch inference

# COMMAND ----------

env_or_alias = "champion" if uc_enabled else "production"
model_info = get_latest_model_version(registered_model_name_fs, env_or_alias)
model_uri = get_model_uri(model_info,env_or_alias)

# COMMAND ----------

model_uri

# COMMAND ----------

#Lets install model dependencies
import mlflow
req_file = mlflow.pyfunc.get_model_dependencies(model_uri)
%pip install -r $req_file

# COMMAND ----------

# MAGIC %md
# MAGIC #### Batch Inference using Feature Store api

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

data = spark.table(inference_data_table_fs)
result_df = fs.score_batch(model_uri, data)

# COMMAND ----------

display(result_df)

# COMMAND ----------


