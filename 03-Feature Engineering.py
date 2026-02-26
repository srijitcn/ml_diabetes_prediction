# Databricks notebook source
# MAGIC %md
# MAGIC #### Feature Engineering in Databricks
# MAGIC
# MAGIC ##### Feature Engineering in Unity Catalog
# MAGIC A feature store is a centralized repository that enables data scientists to find and share features and also ensures that the same code used to compute the feature values is used for model training and inference.
# MAGIC
# MAGIC Machine learning uses existing data to build a model to predict future outcomes. In almost all cases, the raw data requires preprocessing and transformation before it can be used to build a model. This process is called feature engineering, and the outputs of this process are called features - the building blocks of the model.
# MAGIC
# MAGIC Developing features is complex and time-consuming. An additional complication is that for machine learning, feature calculations need to be done for model training, and then again when the model is used to make predictions. These implementations may not be done by the same team or using the same code environment, which can lead to delays and errors. Also, different teams in an organization will often have similar feature needs but may not be aware of work that other teams have done. A feature store is designed to address these problems.
# MAGIC
# MAGIC **Databricks Feature Engineering** (formerly Feature Store) is fully integrated with other components of Databricks:
# MAGIC
# MAGIC - **Discoverability**: The Feature Store UI, accessible from the Databricks workspace, lets you browse and search for existing features.
# MAGIC - **Lineage**: When you create a feature table in Databricks, the data sources used to create the feature table are saved and accessible. For each feature in a feature table, you can also access the models, notebooks, jobs, and endpoints that use the feature.
# MAGIC - **Integration with model scoring and serving**: When you use features from Feature Store to train a model, the model is packaged with feature metadata. When you use the model for batch scoring or online inference, it automatically retrieves features from Feature Store. The caller does not need to know about them or include logic to look up or join features to score new data. This makes model deployment and updates much easier.
# MAGIC - **Point-in-time lookups**: Feature Store supports time series and event-based use cases that require point-in-time correctness.
# MAGIC
# MAGIC **Documentation**:
# MAGIC - [Feature Engineering in Unity Catalog](https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html)
# MAGIC - [Feature Engineering Python API](https://api-docs.databricks.com/python/feature-engineering/latest/index.html)
# MAGIC - [Create a feature table in Unity Catalog](https://docs.databricks.com/en/machine-learning/feature-store/uc/create-feature-table-uc.html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read Input Data and Extract Feature Columns
# MAGIC
# MAGIC Let us create a feature store for our model

# COMMAND ----------

# MAGIC %md
# MAGIC Let us take a look the data tables we have and what we are trying to acheieve

# COMMAND ----------

displayHTML(
   """
   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
   <div class='mermaid'>
    graph TD
      A[patient_demographics]
      B[patient_lab_results]
      C[patient_pysicals]
      D{join}
      Z[diabetes_features]
      E[Patient Registration System] 
      F[Daily Feed Clinical Data]
      G[Daily Batch Physician System]
      E -.-> A
      F -.-> B
      G -.-> C
      A -->|Id| D
      B --> |Id| D
      C --> |Id| D 
      D --> Z
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC Choose "Show Code" in the above cell, to see how to display HTML content in the notebook cell output

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

feature_columns = ["Age", "BloodPressure", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Glucose"]

lab_df = spark.table(lab_results_table)

phys_df = spark.table(physicals_results_table)

feature_data = (spark
                .table(demographic_table)
                .join(lab_df, "Id")
                .join(phys_df,"Id") 
                .select("Id",*feature_columns)              
                )

# COMMAND ----------

display(feature_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Save Data to Feature Table

# COMMAND ----------

# In DBR 17.3+ use FeatureEngineeringClient for Unity Catalog feature tables
# Documentation: https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# Create the feature table in Unity Catalog
# The create_table method creates a Delta table with feature metadata
# Documentation: https://docs.databricks.com/en/machine-learning/feature-store/uc/create-feature-table-uc.html
fe.create_table(
    name=feature_table_name,
    primary_keys="Id",
    df=feature_data,
    description="Diabetes prediction features including patient demographics, lab results, and physical measurements"
)

# COMMAND ----------

# Write additional data to the feature table (merge mode for upserts)
# This is useful for incremental updates to the feature table
fe.write_table(
    name=feature_table_name,
    df=feature_data,
    mode="merge",
)

# COMMAND ----------


