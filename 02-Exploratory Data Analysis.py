# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Exploratory Data Analysis (EDA)
# MAGIC
# MAGIC #### Overview: 
# MAGIC <!-- [Apache Spark](https://www.databricks.com/spark/about) is a powerful open-source processing engine built around: **speed** (*by exploiting in memory computing and other optimizations*), **ease** of use, and sophisticated **analytics**. *`Apache Spark is at the heart of the Databricks Lakehouse Platform and is the technology powering compute clusters and SQL warehouses.`*
# MAGIC *`Databricks is an optimized platform for Apache Spark, providing an efficient and simple platform for running Apache Spark workloads.`* -->
# MAGIC
# MAGIC ----    
# MAGIC
# MAGIC We will demonstrate reading in data using [PySpark DataFrames on Databricks](https://docs.databricks.com/en/getting-started/dataframes-python.html), illustrate how you can visualize the data within Databricks Notebook, and convert our PySpark DataFrame to [Pandas DataFrame](https://www.databricks.com/glossary/pandas-dataframe) for more exploratory data analyses.  
# MAGIC
# MAGIC <!-- **DataFrame**: a spreadsheet, a SQL table, or a dictionary of series objects.    -->
# MAGIC
# MAGIC >Apache Spark DataFrames provide a rich set of functions (`select` `columns`, `filter`, `join`, `aggregate`) 
# MAGIC that allow you to solve common data analysis problems efficiently. 
# MAGIC Apache Spark DataFrames are an abstraction built on top of Resilient Distributed Datasets (RDDs). 
# MAGIC Spark DataFrames and Spark SQL use a unified planning and optimization engine, 
# MAGIC allowing you to get nearly identical performance across all supported languages on Databricks (Python, SQL, Scala, and R).
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC ###0. Data 
# MAGIC
# MAGIC The **Pima Indian Diabetes Dataset** is used for this demo. Pima Indians are a Native American group that lives in Mexico and Arizona, USA [[ref]](https://diabetesjournals.org/care/article/29/8/1866/28611/Effects-of-Traditional-and-Western-Environments-on), and have been indicated to have a high incidence rate of diabetes mellitus. The Pima Indian Diabetes dataset consisting of Pima Indian females 21 years and older is a popular benchmark dataset. 
# MAGIC <!-- [[Original Research]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/) -->
# MAGIC ----    
# MAGIC The raw data consists of the following information:      
# MAGIC
# MAGIC - **`Id`**:long `-- included for feature store look up`     
# MAGIC - **`FirstName`**:string `-- Patient Info.`     
# MAGIC - **`LastName`**:string `-- Patient Info.`     
# MAGIC - **`Address`**:string `-- Patient Info.`     
# MAGIC - **`Email`**:string `-- Patient Info.`     
# MAGIC - **`Age`**:integer `-- Age at study`     
# MAGIC - **`Pregnancies`**:integer `-- Number of times pregnant`     
# MAGIC - **`Glucose`**:integer `-- Plasma glucose concentration a 2 hours in an oral glucose tolerance test`     
# MAGIC - **`BloodPressure`**:integer  `-- Diastolic blood pressure in mm Hg`     
# MAGIC - **`SkinThickness`**:integer `-- Triceps skin fold thickness in mm`     
# MAGIC - **`Insulin`**:integer `-- 2-Hour serum insulin in mu U/ml`     
# MAGIC - **`BMI`**:double `-- Body mass index measured as weight in kg/(height in m)^2`     
# MAGIC - **`DiabetesPedigreeFunction`**:double `-- A diabetes onset likelihood {associated with subject’s age, their diabetic family history, and other factors}`     
# MAGIC - **`Outcome`**:integer `-- Diabetes diagnosis`     
# MAGIC
# MAGIC We will additionally parse out **`State`** information from **`Address`** so we can visualize the distribution of data across the USA.
# MAGIC
# MAGIC ---- 
# MAGIC
# MAGIC <!-- ###1. Read in Data as a PySpark DataFrame  
# MAGIC
# MAGIC ###2. Initial Exploration and minimal processing with PySpark DataFrame
# MAGIC
# MAGIC ###3. EDA with Pandas Profiling
# MAGIC
# MAGIC ###4. Further exploratory analyses with Pandas DataFrame
# MAGIC
# MAGIC ###5. Summary and Recommendations 
# MAGIC
# MAGIC ###6. What's next? -->
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's get started ... 

# COMMAND ----------

# DBTITLE 1,Data 
# MAGIC %run ./init

# COMMAND ----------

# DBTITLE 1,Import Relevant Libraries 
import pyspark.sql.functions as F
import pyspark.sql.types as T

#Importing basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
# from ydata_profiling.utils.cache import cache_file


# COMMAND ----------

# DBTITLE 0,PySpark DataFrames [sdf]
# MAGIC %md
# MAGIC ###1. Read in Data as a PySpark DataFrame  

# COMMAND ----------

## Helper Function

def extract_stateNzip_fromAddress(sdf, getState=True, getZip=False):
  """ 
  Extract State and Zip from Address 
  location can be helpful for deriving a birds-eye-view of how the variables are distributed across the USA
  """
  sdf_out = (sdf           
             .withColumn('_split', F.split('Address',' '))            
             .withColumn('len_split1', F.array_size('_split') )
             .withColumn('State', F.col('_split')[F.col('len_split1')-2])
             .withColumn('Zip', F.col('_split')[F.col('len_split1')-1])                         
             .drop(*['_split','state_zip','len_split1'])
            )
  if getZip:
    return sdf_out.drop('State')
  elif getState and getZip:
    return sdf_out
  else:
    return sdf_out.drop('Zip')


# COMMAND ----------

raw_data_table 

# COMMAND ----------

# DBTITLE 1,Read data 
pima_sdf0 = spark.table(raw_data_table)

pii_cols = ['FirstName', 'LastName','Email','Address']

pima_sdf1 = (extract_stateNzip_fromAddress(pima_sdf0, getState=True, getZip=False)
            #.select(*['state']+pima_sdf0.drop(*pii_cols).columns)
            .select(*pima_sdf0.columns[:4]+['State']+pima_sdf0.columns[4:])
            )

# COMMAND ----------

# DBTITLE 1,Check the Address strings to parse for State
display(pima_sdf0.select('id','Address'))

# COMMAND ----------

# DBTITLE 1,Check the parsed State info
display(pima_sdf1.select('id','Address','State'))

# COMMAND ----------

cols2use = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Outcome']

# COMMAND ----------

display(pima_sdf1.select(*cols2use))

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Initial Exploration and minimal processing with PySpark DataFrame
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Something looks a bit strange in some of the values ... can you spot what it might be? 
dbutils.data.summarize(pima_sdf1[cols2use])

# COMMAND ----------

# DBTITLE 1,Check counts of zeros in feature columns
display(pima_sdf1.select(*[F.sum((F.col(c)==0).cast('integer')).alias(f'{c}_count0') for c in 
                          ['Age', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']]
                        ) 
        )

# COMMAND ----------

# MAGIC %md 
# MAGIC #### NOTE: We will take care of the `"MISSING"` values during Modeling 
# MAGIC ##### -- where we will use [`sklearn.pipelines`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) to handle the [combination of various data transforms](https://scikit-learn.org/stable/modules/compose.html) in data processing and track the steps together with the modeling using [`MLflow`](https://mlflow.org/docs/latest/introduction/index.html)  
# MAGIC ##### However, we need to understand how best to impute and/or transform them.    
# MAGIC

# COMMAND ----------

# DBTITLE 1,Lets define columns with zeros that should be 'missing data' instead 
nonZeroCols=['BloodPressure','SkinThickness','BMI'] #Glucose, Insulin -- levels can get to 0

pima_sdf2=pima_sdf1

## for illustration purpose 
for c in nonZeroCols:
  pima_sdf2 = pima_sdf2.withColumn(c, F.when(F.col(c)!=0, F.col(c).cast(T.DoubleType()) ).otherwise(None))

# COMMAND ----------

# DBTITLE 1,Recheck the counts of zeros 
display(pima_sdf2.select(*[F.sum((F.col(c)==0).cast('integer')).alias(f'{c}_count0') for c in 
                          ['Age', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']]
                        ) 
        )

# COMMAND ----------

display(pima_sdf2.drop(*pii_cols))

# COMMAND ----------

# DBTITLE 1,Example data distributions with notebook viz 
display(pima_sdf2.select(*['State']+cols2use))

# COMMAND ----------

# MAGIC %md 
# MAGIC PySpark is especially great for preprocessing large datasets efficiently and you can easily convert these sparkDF to pandasDF.    
# MAGIC Let's take a look at doing just that and work in a more familiar coding environment. 

# COMMAND ----------

# DBTITLE 0,Pandas DataFrame [pd]
# MAGIC %md
# MAGIC ###3. EDA with Pandas Profiling
# MAGIC

# COMMAND ----------

pima_sdf2.toPandas().describe() #.describe(include='all')

# COMMAND ----------

# DBTITLE 0,EDA using Pandas Profiling | [raw]
#EDA using Pandas Profiling
Preport1 = ProfileReport(pima_sdf1.toPandas()[cols2use],
                        title="Pima Indian Diabetes Dataset [Before Preprocessing]",
                        
                        plot={
                              'correlation':{
                              'cmap': 'RdBu_r',
                              'bad': '#000000'},
                              'fontsize':6,
                             },
                        
                        sort=None, html={'style':{'full_width':True}},
                        # minimal=True
                        )
                     
Preport1.to_notebook_iframe()

# from ydata_profiling.utils.cache import cache_file
# Preport1.to_widgets()


# COMMAND ----------

# MAGIC %md 
# MAGIC NOTE: if the dataset is very large >> Pandas Profiling could take a while.

# COMMAND ----------

# DBTITLE 0,EDA using Pandas Profiling | [replaced 0]
#EDA using Pandas Profiling

Preport2 = ProfileReport(pima_sdf2.toPandas()[cols2use],
                        title="Pima Indian Diabetes Dataset [+ Minimal Preprocessing]",
                        correlations={
                                      "pearson": {"calculate": True},
                                      "spearman": {"calculate": True},
                                      "kendall": {"calculate": True},
                                      "phi_k": {"calculate": True},
                                     },
                        plot={
                              'correlation':{
                              'cmap': 'RdBu_r',
                              'bad': '#000000'},
                              'fontsize':6,
                             },
                        sort=None, html={'style':{'full_width':True}}
                        )
                     
Preport2.to_notebook_iframe()

# from ydata_profiling.utils.cache import cache_file
# Preport2.to_widgets()


# COMMAND ----------

# MAGIC %md
# MAGIC ###4. Further exploratory analyses with Pandas DataFrame
# MAGIC

# COMMAND ----------

# DBTITLE 1,Make pandasDF from sparkDF
pima_pd = pima_sdf2.toPandas() 

# COMMAND ----------

# DBTITLE 1,Get an overview of all the data variables : Boxplots 
sns.set_theme(style="white")
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette('Set2'))

fig = plt.figure(figsize=(8,10))
for i,col in enumerate(set(pima_pd[cols2use].columns)-{"Outcome"}):
  ax = fig.add_subplot(8,1,i+1);
  sns.boxplot(data= pima_pd[col], orient='h', color=cmap.colors[i], width=0.3, ax=ax)
  plt.xticks(fontsize=10);
  plt.ylabel(col, fontsize=10, rotation=0, horizontalalignment='right', labelpad=10);
  plt.yticks(ticks=[],labels=None);

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.suptitle('                               Overview', y=0.925);

# fig.text(0.04, 0.5, 'Variables', va='center', rotation='vertical', fontsize=16)

# COMMAND ----------

# MAGIC %md 
# MAGIC Boxplots help to: 
# MAGIC - Identify outliers or anomalous data points
# MAGIC - To determine if our data is skewed
# MAGIC - To understand the spread/range of the data
# MAGIC
# MAGIC e.g. Insulin : many extreme outliers -- e.g. some std. dev of mean could be used to determine inclusion criteria etc.    
# MAGIC or if Log or other scaling transformation is necessary etc. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Histograms: Data Distributions 
sns.set_theme(style="white")
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette('Set2'))

plt.figure(figsize=(22,28))
plt.subplots_adjust(left=0.2, right=.9, bottom=0.1, top=0.8, wspace=0.4, hspace=0.3)

for i,col in enumerate(set(pima_pd[cols2use].columns)-{"Outcome"}):
  plt.subplot(6,4,i+1);
  sns.histplot(pima_pd[col], 
               color=cmap.colors[i], 
               kde = True, linewidth=1), 
  plt.xticks(fontsize=10);
  plt.xlabel(col, fontsize=15);
  # plt.title(col, fontsize=15);
  # plt.ylabel('Count',fontsize=15)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC NOTES: 
# MAGIC
# MAGIC We observe that the variables `['Age','Insulin','DiabetesPedigreeFunction','Pregnancies','Glucose']` are skewed -- and that if there were missing values in these variables it might be best to impute them with their `median` value.     
# MAGIC On the other hand, variables `['BloodPressure', 'BMI', 'SkinThickness']` are 'approximately' normally distributed  -- and where there are missing values we could impute by their `mean` value.  
# MAGIC
# MAGIC <!-- # ['Age','Insulin','DiabetesPedigreeFunction','Pregnancies','Glucose'] : skewed -- fill "missing" with median
# MAGIC # ['BloodPressure', 'BMI', 'SkinThickness'] : "normal" -- fill "missing" with mean
# MAGIC # >> SimpleImputer / KNN  -->

# COMMAND ----------

# DBTITLE 1,Pairplots : Scatter Plots | Kernel Density Estimate Plots + hue indicated by Outcome
sns.set(font_scale=1.35)

sns.pairplot(data=pima_pd[cols2use],
             hue = 'Outcome', 
             corner=True)
plt.show()

# COMMAND ----------

# DBTITLE 1,Variable Boxplots by Outcomes 
sns.set(font_scale=1)

fig = plt.figure(figsize=(18,12))
plt.subplots_adjust(left=0.2, right=.9, bottom=0.1, top=0.8, wspace=0.4, hspace=0.2)

for i,col in enumerate(set(pima_pd[cols2use].columns)-{'Outcome'}):
    ax = fig.add_subplot(2,4,i+1,)   
    sns.boxplot(data=pima_pd[cols2use],x='Outcome', y=col, ax=ax, width=0.3)
    plt.xlabel('Outcome', fontsize=12)
    plt.xticks(fontsize=10)
    # plt.title(col)
    # plt.ylabel('')

# COMMAND ----------

# MAGIC %md 
# MAGIC - General observation of overal median value differences in the variables for patients that were eventually diagnosed with diabetes compared to those who are not. 
# MAGIC - [Outliers](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm) in each variables for each Outcome category; some more spread out relative to median and IQR, some beyond the Wiskers. 
# MAGIC
# MAGIC SMEs to decide if outliers to be in/excluded, and to rule out if the outlier(s) are   
# MAGIC
# MAGIC - A measurement error or data entry error, correct the error where possible. 
# MAGIC - Not a part of the study population (i.e., unusual properties or conditions)?
# MAGIC - A natural part of the population you are studying? 
# MAGIC
# MAGIC <!-- https://statisticsbyjim.com/basics/remove-outliers/ -->

# COMMAND ----------

# DBTITLE 1,Zoomed into Glucose & Insulin 
sns.jointplot(x="Glucose", y="Insulin", data=pima_pd, hue="Outcome");

# COMMAND ----------

# DBTITLE 1,Zoomed into BMI & SkinThickness 
sns.jointplot(x="BMI", y="SkinThickness", data=pima_pd, hue="Outcome");

# COMMAND ----------

# DBTITLE 1,Pearson Correlation without Pandas Profiling
sns.set_theme(style="white")

# Compute the correlation matrix
corr = pima_pd[cols2use[:-1]].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 5));

cmap = "RdBu_r"

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, 
            mask=mask, cmap=cmap, 
            #vmax=.3, 
            center=0,
            square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}
           );

plt.title("Pearson's Correlation Matrix")

# COMMAND ----------

# MAGIC %md
# MAGIC ###5. Summary and Recommendations 
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **In this walk through, we demonstrated:** 
# MAGIC >- How PySpark DataFrame can help us easily clean, process, and visualize data, which can be advantageous when working with large datasets. 
# MAGIC >- The ease of converting from PySpark DF to Pandas DF (and vice versa) 
# MAGIC >- Ease of profiling data in both in PySpark and Pandas
# MAGIC >- Access to familiar Pandas DF plotting/visualization libraries  e.g seaborn, matplolib etc. to help visualize data distributions, interactions, and outliers.   
# MAGIC  
# MAGIC >- The benefits of SparkDF (*large datasets, speed*) together with the ease and familiarity of practitioners using Pandas DF can also now be leveraged using the [Pandas API on Spark](https://docs.databricks.com/en/pandas/pandas-on-spark.html), which fills the gap of Pandas's big data scaling capabilities by providing `Pandas  equivalent APIs` that work on `Apache Spark`.    
# MAGIC
# MAGIC **In our EDA, we noted the following and their associated handling decision requirements :**      
# MAGIC >- `Missing values` --  Impute decision/strategy e.g. `Mean, Median, KNN`?   
# MAGIC >- `Data skewness` -- Scaling/Transformation e.g. do we sacle/normalize/log-transform ? 
# MAGIC >- `Outliers` -- Include/Exclude basis?   
# MAGIC
# MAGIC
# MAGIC **A residual and important question lingers: How do we go about tracking these decision steps for reproducibility?**    
# MAGIC --> MLflow 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5(i) Quick example of using `sklearn.pipelines` to transform/process `"missing"` data we noted earlier

# COMMAND ----------

# DBTITLE 1,Defining the numerical pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


#Preprocessors
imputers = []
imputers.append(
              ("impute_mean", SimpleImputer(missing_values=0, strategy="mean"), nonZeroCols)
              #("impute_median", SimpleImputer(missing_values=None, strategy="median"), skewedCols) 
              )

numerical_pipeline = Pipeline(steps=[
                                    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))), #converts number-like values >> NaN
                                    ("imputers", ColumnTransformer(imputers)),
                                    # ("standardizer", StandardScaler())
                                    ])

data_transformers = [
                    ("numerical", numerical_pipeline, cols2use)
                    #("categorical", categorical_pipeline, categories2use)
                    ]

preprocessor = ColumnTransformer(data_transformers, remainder="passthrough", sparse_threshold=0)


# COMMAND ----------

#setup MLflow experiment so that it does not create experiment in default path
from datetime import datetime
mlflow.set_registry_uri(model_registry_uri)
experiment_tag = f"{user_prefix}_exploratory_data_analysis_{datetime.now().strftime('%d-%m-%Y')}"
experiment_base_path = f"Users/{user_email}/mlflow_experiments"
dbutils.fs.mkdirs(f"file:/Workspace/{experiment_base_path}")
experiment_path = f"/{experiment_base_path}/{experiment_tag}"
experiment = mlflow.set_experiment(experiment_path)

## we will convert original SparkDF to Pandas to illustrate: 
pima_sdf0_pd = pima_sdf0.select(cols2use).toPandas()

## fit our preprocessor to the original data converted to a pandas df
preprocessor_fitT = preprocessor.fit(pima_sdf0_pd)

## use the fitted preprocessor Transformer to transform the original data 
pima_sdf0_pdT = preprocessor_fitT.transform(pima_sdf0_pd)


# COMMAND ----------

# DBTITLE 0,pima_sdf0_pd [Raw] 
pima_sdf0_pd[nonZeroCols].head(15)

# COMMAND ----------

# DBTITLE 0,pima_sdf0__pdTransformed
pd.DataFrame(pima_sdf0_pdT, columns=nonZeroCols).head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ###6. What's next?

# COMMAND ----------

# MAGIC %md
# MAGIC #### NEXT STEPs: 
# MAGIC - We can run an AutoML with the minimally processed pandas_df 
# MAGIC - Look into Feature Engineering, Hyperparameter Tuning, Data Modeling etc.
# MAGIC
# MAGIC **automl** 
# MAGIC - ref: https://docs.databricks.com/en/machine-learning/automl/how-automl-works.html
# MAGIC - api ref : https://docs.databricks.com/en/machine-learning/automl/train-ml-model-automl-api.html 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run an AutoML 
# import databricks.automl
 
# summary = databricks.automl.classify(pima_pd[cols2use], 
#                                      target_col="Outcome",                                     
#                                      timeout_minutes=6, 
#                                     )
                                    

# COMMAND ----------


