# Databricks notebook source
# MAGIC %md
# MAGIC #### Prepare Data
# MAGIC For this exercise, we will train a LGBM classifier for Diabetes prediction. We will use Feature Engineering with Feature Lookup from Unity Catalog.
# MAGIC
# MAGIC **Runtime Requirements**: DBR 17.3 LTS ML
# MAGIC
# MAGIC **Documentation**:
# MAGIC - [Feature Engineering in Unity Catalog](https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html)
# MAGIC - [Train models with feature tables](https://docs.databricks.com/en/machine-learning/feature-store/train-models-with-feature-store.html)
# MAGIC - [Hyperopt distributed tuning](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

numeric_columns = ["Age", "BloodPressure", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies", "Glucose"]
non_zero_columns = ["BloodPressure", "SkinThickness" , "BMI"]
categorical_columns = []
label_column = "Outcome"
key_column = "Id"

feature_cols = numeric_columns + categorical_columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Features
# MAGIC We start with the table that has the results we want to use and will lookup the features that are missing

# COMMAND ----------

patient_lab_data = spark.table(lab_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Lookup
# MAGIC  A FeatureLookup specifies each feature to use in the training set, including the name of the feature table, the name(s) of the features, and the key(s) to use when joining the feature table with the DataFrame passed to create_training_set. See Feature Lookup for more information.

# COMMAND ----------

displayHTML(
   """
   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
   <div class='mermaid'>
    flowchart LR
      D["`
        **DATA_TABLE**
        id,pid,zid,label
      `"]
      
      T["`
        **TRAINING_DF**
        F1,F2,F3,F4,F5,F6,F7,F8,F9, Label
      `"]
    
     D --> FeatureStoreClient --> T

    subgraph FeatureStoreClient
      F1["`
        **FEATURE_LOOKUP 1**
        id -> F1,F2,F3
      `"]

      F2["`
        **FEATURE_LOOKUP 2**
        pid -> F4,F5,F6
      `"]

      F3["`
        **FEATURE_LOOKUP 3**
        zid -> F7,F8,F9
      `"]
    end
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC Choose "Show Code" in the above cell, to see how to display HTML content in the notebook cell output

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Training Dataset

# COMMAND ----------

# In DBR 17.3+ use FeatureEngineeringClient for Unity Catalog feature tables
# Documentation: https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# FeatureLookup specifies which features to retrieve from the feature table
# The lookup_key is used to join the feature table with the input DataFrame
# Documentation: https://docs.databricks.com/en/machine-learning/feature-store/train-models-with-feature-store.html
feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_names=["Age", "BloodPressure", "BMI", "Pregnancies"],
        lookup_key=[key_column]
    ),
]

# Create a training set that joins features from the feature table with the label data
# The training set tracks which features were used, enabling automatic feature lookup at inference time
training_set = fe.create_training_set(
    df=patient_lab_data,
    feature_lookups=feature_lookups,
    label=label_column,
    exclude_columns=[key_column]
)

training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Train Data Split

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**: The data we used in this exercise is small to fit in driver memory. So we can convert the Spark Dataframe into a Pandas dataframe for ease of use. 
# MAGIC
# MAGIC In case, the training data is large, we should avoid collecting the data to driver and creating a pandas dataframe. Instead, we should use data parellel distributed training. [Learn more about Distributed Training on Databricks](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/index.html)

# COMMAND ----------

from sklearn.model_selection import train_test_split

training_df_pd = training_df.toPandas()
y = training_df_pd[label_column]
x = training_df_pd[feature_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Model
# MAGIC
# MAGIC We will use an LGBM CLassifier for this exercise. LGBMClassifier stands for Light Gradient Boosting Machine Classifier. It uses decision tree algorithms for ranking, classification, and other machine-learning tasks. LGBMClassifier uses a novel technique of Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to handle large-scale data with accuracy, effectively making it faster and reducing memory usage.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import make_column_selector as selector
import pandas as pd

import lightgbm
from lightgbm import LGBMClassifier

def get_model(model_params):
  #Preprocessors
  imputers = []
  imputers.append(
    ("impute_mean", SimpleImputer(missing_values=0), non_zero_columns)
  )

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
      ("imputers", ColumnTransformer(imputers)),
      ("standardizer", StandardScaler())
  ])

  numerical_transformers = [
    ("numerical", numerical_pipeline, numeric_columns)
  ]

  #since we have only numerical transformation, we can diretcly use `numerical_transformers` 
  preprocessor = ColumnTransformer(numerical_transformers, remainder="passthrough", sparse_threshold=0)

  #Model
  lgbmc_classifier = LGBMClassifier(**model_params)

  #Pipeline
  model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgbmc_classifier),
    ])
  
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC #### HyperOpt library
# MAGIC Databricks Runtime ML includes Hyperopt, a Python library that facilitates distributed hyperparameter tuning and model selection. With Hyperopt, you can scan a set of Python models while varying algorithms and hyperparameters across spaces that you define. Hyperopt works with both distributed ML algorithms such as Apache Spark MLlib and Horovod, as well as with single-machine ML models such as scikit-learn and TensorFlow.
# MAGIC
# MAGIC The basic steps when using Hyperopt are:
# MAGIC
# MAGIC - Define an objective function to minimize. Typically this is the training or validation loss.
# MAGIC - Define the hyperparameter search space. Hyperopt provides a conditional search space, which lets you compare different ML algorithms in the same run.
# MAGIC - Specify the search algorithm. Hyperopt uses stochastic tuning algorithms that perform a more efficient search of hyperparameter space than a deterministic grid search.
# MAGIC - Run the Hyperopt function fmin(). fmin() takes the items you defined in the previous steps and identifies the set of hyperparameters that minimizes the objective function.
# MAGIC
# MAGIC Read more about
# MAGIC - [HyperOpt Concepts](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)
# MAGIC - [Use distributed training algorithms with Hyperopt](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html)
# MAGIC - [Best practices: Hyperparameter tuning with Hyperopt](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Objective Function
# MAGIC

# COMMAND ----------

import mlflow
import os
from datetime import datetime 
from sklearn.metrics import f1_score,roc_auc_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import numpy as np

mlflow.set_registry_uri(model_registry_uri)

#Databricks Utilities (dbutils): https://docs.databricks.com/en/dev-tools/databricks-utils.html
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

#Create an MLFlow experiment
experiment_tag = f"{user_prefix}_diabetes_prediction_fs_{datetime.now().strftime('%d-%m-%Y')}"
experiment_base_path = f"Users/{user_email}/mlflow_experiments"
dbutils.fs.mkdirs(f"file:/Workspace/{experiment_base_path}")
experiment_path = f"/{experiment_base_path}/{experiment_tag}"

# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)

def loss_fn(params):

  # Initialize MLFlow
  # Remember this function will run on workers
  os.environ['DATABRICKS_HOST'] = db_host   
  os.environ['DATABRICKS_TOKEN'] = db_token   
  
  with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_run:
    #enable sklearn autologging
    mlflow.sklearn.autolog(disable=True,
                           log_input_examples=True,
                           silent=True)
    
    #get the model
    model = get_model(params)

    #Now lets train the model
    model.fit(x_train,y_train)

    #evaluate the model
    preds = model.predict(x_test)

    #the metric we want to minimize
    roc_score = roc_auc_score(preds,y_test)
    f1score = f1_score(preds,y_test)
    
    mlflow.log_metric("roc_score",roc_score)
    mlflow.log_metric("f1score",f1score)
    mlflow.sklearn.log_model(model,artifact_path="model")

    return {"loss": -f1score,"status": STATUS_OK}


# COMMAND ----------

algo=tpe.suggest

#Reference: https://hyperopt.github.io/hyperopt/getting-started/search_spaces/
search_space = {
  "colsample_bytree": hp.uniform("colsample_bytree",0,1),
  "lambda_l1": hp.uniform("lambda_l1",0,0.5), 
  "lambda_l2": hp.uniform("lambda_l2",0,0.5), 
  "learning_rate": hp.lognormal("learning_rate",0, 1), 
  "max_bin": hp.choice('max_bin', np.arange(50, 255, dtype=int)), 
  "max_depth": hp.choice('max_depth', np.arange(10, 20, dtype=int)), 
  "min_child_samples": 35,
  "n_estimators": 181,
  "num_leaves": 90,
  "path_smooth": 82.58514740065912,
  "subsample": 0.7664087623951591,
  "random_state": 50122439,
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### HyperParameter Tuning and Training

# COMMAND ----------

trials = SparkTrials()
fmin(
  fn=loss_fn,
  space=search_space,
  algo=algo,
  max_evals=5,
  trials=trials)


# COMMAND ----------

best_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=['metrics.f1score DESC']).iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model

# COMMAND ----------

model_uri = f"runs:/{best_run.run_id}/model"
selected_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

from sklearn.model_selection import cross_val_score
score = cross_val_score(selected_model, x_test, y_test, cv=5, scoring='f1')
print(f"F1 score of mode is {score}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot Model Performance

# COMMAND ----------

# Visualization of model performance metrics
# Note: In scikit-learn 1.2+ (included in DBR 17.3), plot_* functions are replaced with Display classes
# Documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#visualizations
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay

fig1, axs1 = plt.subplots(1)
RocCurveDisplay.from_estimator(selected_model, x_test, y_test, ax=axs1)
plt.title("ROC Curve")
plt.show()

# COMMAND ----------

fig2, axs2 = plt.subplots(1)
ConfusionMatrixDisplay.from_estimator(selected_model, x_test, y_test, ax=axs2)
plt.title("Confusion Matrix")
plt.show()

# COMMAND ----------

fig3, axs3 = plt.subplots(1)
PrecisionRecallDisplay.from_estimator(selected_model, x_test, y_test, ax=axs3)
plt.title("Precision-Recall Curve")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Model
# MAGIC Now that we have a good confidence in the model, let us register in the model registry and promote the model to Staging for integration testing

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(x_test,y_test)

# COMMAND ----------

# Log the model with feature metadata using FeatureEngineeringClient
# This enables automatic feature lookup at inference time
# Documentation: https://docs.databricks.com/en/machine-learning/feature-store/train-models-with-feature-store.html#log-model
with mlflow.start_run() as run:
    # Log model with feature engineering metadata
    # Note: pip_requirements updated to match DBR 17.3 LTS ML library versions
    fe.log_model(
        model=selected_model,
        artifact_path="model",
        flavor=mlflow.sklearn,    
        training_set=training_set,
        registered_model_name=registered_model_name_fs,
        pip_requirements=["lightgbm==4.6.0", "scikit-learn==1.6.1"]
    )
    
    # Evaluate model using MLflow's built-in evaluator
    eval_data = x_test.copy()
    eval_data["target"] = y_test
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"]
    )

    # Log hyperparameters from the best run
    run_data = mlflow.get_run(best_run.run_id).data.to_dictionary()
    mlflow.log_params(run_data["params"])

    # Log visualization artifacts
    mlflow.log_figure(fig1, 'sklearn_roc_curve.png')
    mlflow.log_figure(fig2, 'sklearn_confusion_matrix.png')
    mlflow.log_figure(fig3, 'sklearn_precision_recall_curve.png')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote Model to Staging
# MAGIC ##### Non Unity Catalog Only
# MAGIC Workspace MLflow Model Registry defines several model stages: None, Staging, Production, and Archived. Each stage has a unique meaning. For example, Staging is meant for model integration testing, while Production is for models that have completed the testing or review processes and have been deployed to applications.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

if not uc_enabled:
  mlflow_client = MlflowClient()
  model_details = get_latest_model_version(registered_model_name_fs,"None")
  result = mlflow_client.transition_model_version_stage(name=registered_model_name_fs,
                                        version=model_details.version,
                                        stage="Staging",
                                        archive_existing_versions=True)
  
  print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unity Catalog Only
# MAGIC We will use aliases in case Unity Catalog Managed model registry

# COMMAND ----------

if uc_enabled:
  mlflow_client = MlflowClient()
  model_details = get_latest_model_version(registered_model_name_fs)  
  result = mlflow_client.set_registered_model_alias(registered_model_name_fs,
                                                    version=model_details.version, 
                                                    alias="challenger")
