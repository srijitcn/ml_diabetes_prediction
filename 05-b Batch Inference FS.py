# Databricks notebook source
# MAGIC %md
# MAGIC #### Batch Inference with Feature Engineering (Feature Store)
# MAGIC This notebook demonstrates batch inference using Feature Engineering in Unity Catalog. The model was logged with feature metadata using `fe.log_model()`, so feature lookup happens automatically during scoring.
# MAGIC
# MAGIC **Key Benefits**:
# MAGIC - Automatic feature lookup from feature tables
# MAGIC - Consistent feature transformations between training and inference
# MAGIC - No need to manually join feature data
# MAGIC
# MAGIC **Runtime Requirements**: DBR 17.3 LTS ML
# MAGIC
# MAGIC **Documentation**:
# MAGIC - [Batch inference with Feature Engineering](https://docs.databricks.com/en/machine-learning/feature-store/inference-with-feature-store.html)
# MAGIC - [score_batch API](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html#databricks.feature_engineering.client.FeatureEngineeringClient.score_batch)

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Batch Inference Data
# MAGIC Now we will create some mock data to use for inference. 
# MAGIC
# MAGIC Since we are using Feature Engineering, the input DataFrame only needs the lookup keys and any features not in the feature table. The remaining features will be automatically retrieved from the feature table.

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
# MAGIC #### Batch Inference using Feature Engineering API
# MAGIC The `score_batch` method automatically:
# MAGIC 1. Looks up features from the feature table using the lookup keys (Id)
# MAGIC 2. Joins the features with the input DataFrame
# MAGIC 3. Applies the model to make predictions
# MAGIC
# MAGIC **Documentation**: [score_batch API](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html#databricks.feature_engineering.client.FeatureEngineeringClient.score_batch)

# COMMAND ----------

# Use FeatureEngineeringClient for Unity Catalog feature tables (DBR 17.3+)
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Load the inference data - only needs Id and features not in the feature table
data = spark.table(inference_data_table_fs)

# score_batch automatically retrieves features from the feature table and makes predictions
# The model was logged with fe.log_model() which captured the feature lookup metadata
result_df = fe.score_batch(model_uri, data)

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Alternative: Using pandas_udf with Feature Store for Custom Processing
# MAGIC For scenarios where you need custom post-processing or want to use pandas_udf with feature lookup,
# MAGIC you can first use `score_batch` and then apply additional transformations using `mapInPandas`.
# MAGIC
# MAGIC **Documentation**: [Pandas UDFs](https://docs.databricks.com/en/udf/pandas.html)

# COMMAND ----------

from typing import Iterator
import pandas as pd
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, DoubleType, StringType

# Example: Add a risk category based on prediction probability
def add_risk_category(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for batch in batches:
        # Add custom post-processing: categorize predictions into risk levels
        batch["risk_category"] = batch["prediction"].apply(
            lambda x: "High Risk" if x == 1 else "Low Risk"
        )
        yield batch

# Define output schema with the additional risk_category column
output_schema_with_risk = result_df.schema.add(StructField("risk_category", StringType()))

# Apply custom pandas processing on top of feature store predictions
result_with_risk = result_df.mapInPandas(add_risk_category, schema=output_schema_with_risk)
display(result_with_risk)

# COMMAND ----------


