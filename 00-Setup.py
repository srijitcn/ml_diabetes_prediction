# Databricks notebook source
# MAGIC %md
# MAGIC #### Compute
# MAGIC Use a multi-node cluster with **DBR 17.3 LTS ML** Runtime and 2-4 worker nodes.
# MAGIC
# MAGIC **Documentation**:
# MAGIC - [Databricks Runtime 17.3 LTS ML](https://docs.databricks.com/en/release-notes/runtime/17.3lts-ml.html)
# MAGIC - [Cluster configuration best practices](https://docs.databricks.com/en/compute/cluster-config-best-practices.html) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Setup
# MAGIC - If Unity catalog is not enabled, the datafiles need to be manually copied to an S3 folder/ADLS Container location or dbfs location
# MAGIC - Copy the `diabetes.csv` file
# MAGIC - Copy the `Postural_Tremor_DA_Raw.csv` file
# MAGIC - If Data is in S3/ADLS, an IAM role or Azure Service Principal need to be created and configured
# MAGIC - If UC is enabled, we recommend using a Volume for storing the raw data
# MAGIC - If UC is not enabled, Create a cluster with the instance profile/service principal with the above IAM role
# MAGIC - Test access

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delta Tables

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Database

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Data Tables

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {demographic_table}")
spark.sql(f"DROP TABLE IF EXISTS {lab_results_table}")
spark.sql(f"DROP TABLE IF EXISTS {physicals_results_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Feature Tables

# COMMAND ----------

# Drop the delta table
spark.sql(f"DROP TABLE IF EXISTS {feature_table_name}")

# Remove the feature table entry using Feature Engineering Client
# Documentation: https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html
# Note: In DBR 17.3+, use FeatureEngineeringClient for Unity Catalog feature tables
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
try:
  # Check if feature table exists. Delete if exists
  fe.drop_table(name=feature_table_name)
except:
  print(f"Feature table {feature_table_name} not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cleanup Inference Tables

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_nonfs}")
spark.sql(f"DROP TABLE IF EXISTS {inference_data_table_fs}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Volume and Copy Raw Data

# COMMAND ----------

if volume_raw_data != "uc_not_available":
  spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_raw_data}")
  dbutils.fs.cp(f"file:/Workspace{project_root_path}/_resources/data",raw_data_path,True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copy Images for Markdown

# COMMAND ----------

src_folder = f"file:/Workspace{project_root_path}/_resources/images"
tgt_folder = "/FileStore/tmp/diabetes_prediction"

# COMMAND ----------

dbutils.fs.rm(tgt_folder,True)
dbutils.fs.mkdirs(tgt_folder)
dbutils.fs.cp(src_folder,tgt_folder, True)

# COMMAND ----------

dbutils.fs.ls(tgt_folder)

# COMMAND ----------


