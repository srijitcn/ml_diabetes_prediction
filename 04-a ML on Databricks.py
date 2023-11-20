# Databricks notebook source
# MAGIC %md
# MAGIC #### Machine Learning in Databricks
# MAGIC Built on an open lakehouse architecture, AI and Machine Learning on Databricks empowers ML teams to prepare and process data, streamlines cross-team collaboration and standardizes the full ML lifecycle from experimentation to production including for generative AI and large language models.
# MAGIC
# MAGIC <img src="files/tmp/diabetes_prediction/mldbx.webp" width="500">
# MAGIC
# MAGIC **Auto ML**: Databricks AutoML allows you to quickly generate baseline models and notebooks. [Read More](https://www.databricks.com/product/automl)
# MAGIC
# MAGIC **Distributed Hyper Parameter Tuning**: Databricks Runtime ML includes Hyperopt, a Python library that facilitates distributed hyperparameter tuning and model selection. With Hyperopt, you can scan a set of Python models while varying algorithms and hyperparameters across spaces that you define. [Read More](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/index.html)
# MAGIC
# MAGIC **Distributed Training**: Databricks enables distributed training and inference if your model or your data are too large to fit in memory on a single machine. For these workloads, Databricks Runtime ML includes the TorchDistributor, Horovod and spark-tensorflow-distributor packages. [Read More](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/index.html)
# MAGIC
# MAGIC **Automate experiment tracking and governance**: Managed MLflow automatically tracks your experiments and logs parameters, metrics, versioning of data and code, as well as model artifacts with each training run. You can quickly see previous runs, compare results and reproduce a past result, as needed. Once you have identified the best version of a model for production, register it to the Model Registry to simplify handoffs along the deployment lifecycle. [Read More](https://www.databricks.com/product/managed-mlflow)
# MAGIC
# MAGIC **Manage the full model lifecycle from data to production — and back**: Once trained models are registered, you can collaboratively manage them through their lifecycle with the Model Registry. Models can be versioned and moved through various stages, like experimentation, staging, production and archived. The lifecycle management integrates with approval and governance workflows according to role-based access controls. Comments and email notifications provide a rich collaborative environment for data teams. 
# MAGIC - [Read More About Model Lifecycle Management in a Non UC Workspace](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/workspace-model-registry.html)
# MAGIC - [Read More About Model Lifecycle Management in a UC Enabled Workspace](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)
# MAGIC
# MAGIC **Deploy ML models at scale and low latency**: Deploy models with a single click without having to worry about server management or scale constraints. With Databricks, you can deploy your models as REST API endpoints anywhere with enterprise-grade availability.
# MAGIC - [Deployment for Batch and Streaming Inference](https://docs.databricks.com/en/machine-learning/model-inference/index.html)
# MAGIC - [Deployment for Real Time Inference - Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
# MAGIC - [Model deployment patterns](https://docs.databricks.com/en/machine-learning/mlops/deployment-patterns.html)
# MAGIC
# MAGIC **Use generative AI and large language models**:
# MAGIC Integrate existing pretrained models — such as those from the Hugging Face transformers library or other open source libraries — into your workflow. Transformer pipelines make it easy to use GPUs and allow batching of items sent to the GPU for better throughput. 
# MAGIC
# MAGIC Customize a model on your data for your specific task. With the support of open source tooling, such as Hugging Face and DeepSpeed, you can quickly and efficiently take a foundation LLM and start training with your own data to have more accuracy for your domain and workload. This also gives you control to govern the data used for training so you can make sure you’re using AI responsibly. [Read More](https://docs.databricks.com/en/large-language-models/index.html#:~:text=Databricks%20makes%20it%20simple%20to,source%20libraries%20into%20your%20workflow.)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Developer Workflow
# MAGIC
# MAGIC <img src="files/tmp/diabetes_prediction/devworkflow.png">

# COMMAND ----------

# MAGIC %md 
# MAGIC #### MLOps on Databricks
# MAGIC
# MAGIC MLOps is a set of processes and automated steps to manage code, data, and models. It combines DevOps, DataOps, and ModelOps.
# MAGIC
# MAGIC ML assets such as code, data, and models are developed in stages that progress from early development stages that do not have tight access limitations and are not rigorously tested, through an intermediate testing stage, to a final production stage that is tightly controlled. The Databricks Lakehouse platform lets you manage these assets on a single platform with unified access control. You can develop data applications and ML applications on the same platform, reducing the risks and delays associated with moving data around. [Read more about MLOps Workflow on Databricks](https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Lifecycle
# MAGIC
# MAGIC ###### With UC 
# MAGIC <img src="files/tmp/diabetes_prediction/mlops.png" >
# MAGIC
# MAGIC ###### Without UC
# MAGIC The difference in the workflow without UC is 
# MAGIC 1) Model is tracked via Workspace Model registry instead of Unity Catalog
# MAGIC 2) Instead of Model Aliases we will use Stages. From `None` -> `Staging` -> `Production` -> `Archived`
# MAGIC
# MAGIC **The Big Book of MLOps** covers how to collaborate on a common platform using powerful, open frameworks such as Delta Lake for data pipelines, MLflow for model management (including LLMs) and Databricks Workflows for automation. [Get The Big Book of MLOps](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)

# COMMAND ----------


