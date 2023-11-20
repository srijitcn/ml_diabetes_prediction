# Machine Learning on Databricks 
This repository will have code files for a complete Data Science project on Databricks platform. This example try to solve a classification problem using the Pima Indian Diabetes dataset.

#### Usage:
##### Using Databricks Repos
1) Open your Databricks Workspace. Create a compute with Databricks Runtime `13.3 LTS ML` or later.
2) Clone this repo ([Read How](https://docs.databricks.com/en/repos/git-operations-with-repos.html#clone-a-repo-connected-to-a-remote-repo))

   **NOTE** If you are trying to connect to a repo which is access controlled, configure Github integration first ([Read how](https://docs.databricks.com/en/repos/get-access-tokens-from-git-provider.html))
3) Run the `00-Setup` Notebook first. This need to be only run once
4) Run the `01-Data Ingestion` Notebook next. This need to be only run once
5) Run the notebooks `02-Exploratory Data Analysis`, `03-Feature Engineering`, `04-b HPT Training Evaluation NoFS` and `04-c HPT Training Evaluation FS` in the order. 
 
 **NOTE:**`04-b,c` notebooks registers model in Model Registry or Unity Catalog based your Workspace status. Before running any other notebooks, please make sure the registered models are marked to be used for Production. 
 - If Unity Catalog is enabed, make sure the model version has the `champion` alias. [Read How](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)
 - If Unity Catalog is not enabled, Use model registry UI to Transition the version to `Production`[Read How](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/workspace-model-registry.html)
6) Any time you want to cleanup all the resources you created, run the `09-Cleanup` Notebook. You can always recreate the resources by re-running the notebooks from Step 3
##### Import via zip file
If you dont have access to public repos in your workspace, you can follow the below steps
1) Download the project as a zip file by navigating to `<> Code` and selecting `Download ZIP` from the `Code` Dropdown
2) In your Databricks workspace, import the zip file using `Import` option from the menu bar. ([Read How](https://docs.databricks.com/en/notebooks/notebook-export-import.html#import-a-notebook))

#### Overview of ML on Databricks:
Built on an open lakehouse architecture, AI and Machine Learning on Databricks empowers ML teams to prepare and process data, streamlines cross-team collaboration and standardizes the full ML lifecycle from experimentation to production including for generative AI and large language models.

<img src="https://www.databricks.com/en-website-assets/static/a5e612c6d98dd3ea3948b13c1f3b2d41/03b97/Machine-Learning-graphic-1%20(1).webp" width="500">

**Auto ML**: Databricks AutoML allows you to quickly generate baseline models and notebooks. [Read More](https://www.databricks.com/product/automl)

**Distributed Hyper Parameter Tuning**: Databricks Runtime ML includes Hyperopt, a Python library that facilitates distributed hyperparameter tuning and model selection. With Hyperopt, you can scan a set of Python models while varying algorithms and hyperparameters across spaces that you define. [Read More](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/index.html)

**Distributed Training**: Databricks enables distributed training and inference if your model or your data are too large to fit in memory on a single machine. For these workloads, Databricks Runtime ML includes the TorchDistributor, Horovod and spark-tensorflow-distributor packages. [Read More](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/index.html)

**Automate experiment tracking and governance**: Managed MLflow automatically tracks your experiments and logs parameters, metrics, versioning of data and code, as well as model artifacts with each training run. You can quickly see previous runs, compare results and reproduce a past result, as needed. Once you have identified the best version of a model for production, register it to the Model Registry to simplify handoffs along the deployment lifecycle. [Read More](https://www.databricks.com/product/managed-mlflow)

**Manage the full model lifecycle from data to production — and back**: Once trained models are registered, you can collaboratively manage them through their lifecycle with the Model Registry. Models can be versioned and moved through various stages, like experimentation, staging, production and archived. The lifecycle management integrates with approval and governance workflows according to role-based access controls. Comments and email notifications provide a rich collaborative environment for data teams. 
- [Read More About Model Lifecycle Management in a Non UC Workspace](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/workspace-model-registry.html)
- [Read More About Model Lifecycle Management in a UC Enabled Workspace](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)

**Deploy ML models at scale and low latency**: Deploy models with a single click without having to worry about server management or scale constraints. With Databricks, you can deploy your models as REST API endpoints anywhere with enterprise-grade availability.
- [Deployment for Batch and Streaming Inference](https://docs.databricks.com/en/machine-learning/model-inference/index.html)
- [Deployment for Real Time Inference - Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [Model deployment patterns](https://docs.databricks.com/en/machine-learning/mlops/deployment-patterns.html)

**Use generative AI and large language models**:
Integrate existing pretrained models — such as those from the Hugging Face transformers library or other open source libraries — into your workflow. Transformer pipelines make it easy to use GPUs and allow batching of items sent to the GPU for better throughput. 

Customize a model on your data for your specific task. With the support of open source tooling, such as Hugging Face and DeepSpeed, you can quickly and efficiently take a foundation LLM and start training with your own data to have more accuracy for your domain and workload. This also gives you control to govern the data used for training so you can make sure you’re using AI responsibly. [Read More](https://docs.databricks.com/en/large-language-models/index.html#:~:text=Databricks%20makes%20it%20simple%20to,source%20libraries%20into%20your%20workflow.)
