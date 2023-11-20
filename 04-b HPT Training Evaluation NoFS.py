# Databricks notebook source
# MAGIC %md
# MAGIC #### Prepare Data
# MAGIC
# MAGIC For this exercise, we will train a LGBM classfier for Diabetes prediction. We will not be using Feature Store

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
# MAGIC Since we are not using Feature Store in this example, we need to join the appropriate tables to get the features for our training

# COMMAND ----------

lab_df = spark.table(lab_results_table)

phys_df = spark.table(physicals_results_table)

feature_data = (spark
                .table(demographic_table)
                .join(lab_df, "Id")
                .join(phys_df,"Id") 
                .select("Id",*(feature_cols + [label_column] ))
                )

# COMMAND ----------

display(feature_data)

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

training_df_pd = feature_data.toPandas()
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

model_registry_uri

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
experiment_tag = f"{user_prefix}_diabetes_prediction_nofs_{datetime.now().strftime('%d-%m-%Y')}"
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

import matplotlib.pyplot as plt
from sklearn import metrics

fig1, axs1 = plt.subplots(1)
image_roc_curve = metrics.plot_roc_curve(selected_model, x_test, y_test,ax=axs1)
plt.show()

# COMMAND ----------

fig2, axs2 = plt.subplots(1)
image_confusion_matrix = metrics.plot_confusion_matrix(selected_model, x_test, y_test,ax=axs2)
plt.show()

# COMMAND ----------

fig3, axs3 = plt.subplots(1)
image_precision_recall_curve = metrics.plot_precision_recall_curve(selected_model, x_test, y_test,ax=axs3)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Model
# MAGIC Now that we have a good confidence in the model, let us register in the model registry and promote the model to Staging for integration testing

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(x_test,y_test)

# COMMAND ----------

input_example = {
    "Age":31.0,
    "BloodPressure":66.0,
    "Insulin":0.0,
    "BMI":26.6,
    "SkinThickness":29.0,
    "DiabetesPedigreeFunction":0.351,
    "Pregnancies":1.0,
    "Glucose":85.0
    }

with mlflow.start_run() as run:

    model_info = mlflow.sklearn.log_model(
        selected_model,
        signature = signature,
        artifact_path="model",
        registered_model_name=registered_model_name_non_fs,
        input_example=input_example,
        pip_requirements = ["lightgbm==3.3.5","scikit-learn==1.1.1"]
    )

    eval_data = x_test
    eval_data["target"] = y_test
    result = mlflow.evaluate(
        model_info.model_uri,
        eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"]
    )

    run_data = mlflow.get_run(best_run.run_id).data.to_dictionary()
    mlflow.log_params(run_data["params"])

    mlflow.log_figure(fig1, 'sklearn_roc_curve.png')
    mlflow.log_figure(fig2, 'sklearn_confusion_matrix.png')
    mlflow.log_figure(fig3, 'sklearn_precision_recall_curve.png')


# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote Model to Staging

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Non Unity Catalog Only
# MAGIC Workspace MLflow Model Registry defines several model stages: None, Staging, Production, and Archived. Each stage has a unique meaning. For example, Staging is meant for model integration testing, while Production is for models that have completed the testing or review processes and have been deployed to applications.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

if not uc_enabled:
  mlflow_client = MlflowClient()
  model_details = get_latest_model_version(registered_model_name_non_fs,"None")
  result = mlflow_client.transition_model_version_stage(name=registered_model_name_non_fs,
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
  model_details = get_latest_model_version(registered_model_name_non_fs)
  result = mlflow_client.set_registered_model_alias(registered_model_name_non_fs,
                                                    version=model_details.version, 
                                                    alias="challenger")

# COMMAND ----------


