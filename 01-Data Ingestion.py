# Databricks notebook source
# MAGIC %md
# MAGIC #### Data Ingestion with Databricks
# MAGIC Databricks offers a variety of ways to help you load data into a lakehouse backed by Delta Lake. With Databricks, you can ingest data from hundreds of data sources incrementally and efficiently into your Delta Lake to ensure your lakehouse always contains the most complete and up-to-date data available for data science, machine learning and business analytics. [Read more](https://docs.databricks.com/en/ingestion/index.html)
# MAGIC
# MAGIC ##### Auto Loader
# MAGIC Use Auto Loader to ingest any file that can land in a data lake into Delta Lake. Point Auto Loader to a directory on cloud storage services like Amazon S3, Azure Data Lake Storage or Google Compute Storage, and Auto Loader will incrementally process new files with exactly once semantics.[Read More](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
# MAGIC
# MAGIC Few more useful tutorials are given below
# MAGIC - [Tutorial: Run your first ETL workload on Databricks](https://docs.databricks.com/en/getting-started/etl-quick-start.html)
# MAGIC - [Load data using streaming tables (Python/SQL notebook)](https://docs.databricks.com/en/ingestion/onboard-data.html)
# MAGIC - [Load data using streaming tables in Databricks SQL](https://docs.databricks.com/en/sql/load-data-streaming-table.html)
# MAGIC
# MAGIC ##### Streaming Data Sources
# MAGIC With Databricks, you can pull data from popular message queues, such as Apache Kafka, Azure Event Hubs or AWS Kinesis at lower latencies. By ingesting your data from these sources into your Delta Lake, you don’t have to worry about losing data within these services due to retention policies. You can reprocess data cheaper and more efficiently as business requirements evolve, and you can keep a longer historical view of your data to power machine learning as well as business analytics applications.
# MAGIC To learn more about specific configurations for streaming from or to message queues, see:
# MAGIC - [Kafka](https://docs.databricks.com/en/structured-streaming/kafka.html)
# MAGIC - [Kinesis](https://docs.databricks.com/en/structured-streaming/kinesis.html)
# MAGIC - [Event Hubs](https://docs.databricks.com/en/structured-streaming/streaming-event-hubs.html)
# MAGIC - [Azure Synapse with Structured Streaming](https://docs.databricks.com/en/structured-streaming/synapse.html)
# MAGIC - [Pub/Sub](https://docs.databricks.com/en/structured-streaming/pub-sub.html)
# MAGIC - [Pulsar](https://docs.databricks.com/en/structured-streaming/pulsar.html)
# MAGIC
# MAGIC ##### JDBC DataSources
# MAGIC You can use JDBC to connect with many data sources. Databricks Runtime includes drivers for a number of JDBC databases, but you might need to install a driver or different driver version to connect to your preferred database. Supported databases include the following:
# MAGIC
# MAGIC - [Query PostgreSQL with Databricks](https://docs.databricks.com/en/external-data/postgresql.html)
# MAGIC - [Query MySQL with Databricks](https://docs.databricks.com/en/external-data/mysql.html)
# MAGIC - [Query MariaDB with Databricks](https://docs.databricks.com/en/external-data/mariadb.html)
# MAGIC - [Query SQL Server with Databricks](https://docs.databricks.com/en/external-data/sql-server.html)
# MAGIC - [Use the Databricks connector to connect to another Databricks workspace](https://docs.databricks.com/en/external-data/databricks.html)
# MAGIC
# MAGIC You may prefer Lakehouse Federation for managing queries to external database systems. See [Run queries using Lakehouse Federation](https://docs.databricks.com/en/query-federation/index.html).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Setup 
# MAGIC For this exercise, we will load the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The dataset is downloaded as a csv file and loaded into an S3 location. 
# MAGIC
# MAGIC We will read it from the S3 location and add some fake demographic information and split the data into 3 tables to showcase Feature Tables

# COMMAND ----------

#Will use Faker library to create some fake demographic information
%pip install Faker 

# COMMAND ----------

#dbutils - Databricks Utilities documentation https://docs.databricks.com/en/dev-tools/databricks-utils.html
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ingest Data

# COMMAND ----------

data_file_name = f"{raw_data_path}/diabetes.csv"

# COMMAND ----------

from faker import Faker
faker = Faker()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Pyspark User Defined Functions (UDF)
# MAGIC PySpark UDF are custom functions that defines data manipulation operations on a DataFrame. Databricks has support for many different types of UDFs to allow for distributing extensible logic. [Read more about User Defined Functions](https://docs.databricks.com/en/udf/index.html)
# MAGIC

# COMMAND ----------

# DBTITLE 0,Create UDF for faking demographic information
@udf
def create_first_name():
  return faker.first_name()

@udf
def create_last_name():
  return faker.last_name()

@udf
def create_address():
  return faker.address()

@udf
def create_email():
  return faker.email()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### PySpark DataFrame Functions
# MAGIC A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. You can think of a DataFrame like a spreadsheet, a SQL table, or a dictionary of series objects. Apache Spark DataFrames provide a rich set of functions (select columns, filter, join, aggregate) that allow you to solve common data analysis problems efficiently.
# MAGIC
# MAGIC You can load and transform data using the Apache Spark Python (PySpark) DataFrame API in Databricks. See [Apache Spark PySpark API reference](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrame.html).

# COMMAND ----------

# DBTITLE 0,Apply UDF on PySpark DataFrame
from pyspark.sql.functions import monotonically_increasing_id

df = (spark
      .read
      .option("header","true")
      .option("inferSchema","true")
      .csv(data_file_name)
      .withColumn("Id", monotonically_increasing_id())
      .withColumn("FirstName",create_first_name())
      .withColumn("LastName",create_last_name())
      .withColumn("Address",create_address())
      .withColumn("Email",create_email())
      .select("Id",
              "FirstName",
              "LastName",
              "Address",
              "Email",
              "Age",
              "Pregnancies",
              "Glucose",
              "BloodPressure",
              "SkinThickness",
              "Insulin",
              "BMI",
              "DiabetesPedigreeFunction",
              "Outcome"
              )
)


# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write Delta Tables
# MAGIC
# MAGIC For this exercise we will create three data tables as shown below

# COMMAND ----------

displayHTML(
   """
   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
   <div class='mermaid'>
    erDiagram
    patient_demographics ||--o{ patient_lab_results : has
    patient_demographics ||--o{ patient_pysicals : has
    patient_demographics {
        int Id
        string FirstName
        string LastName
        string Address
        string Email
        string Age
    }
    
    patient_lab_results {
        int Id
        int Glucose
        decimal SkinThickness
        int Insulin
        decimal DiabetesPedigreeFunction
        int Outcome
    }
    patient_pysicals {
        int Id
        int Pregnancies
        int BloodPressure
        decimal BMI
    }
    </div>
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC Choose "Show Code" in the above cell, to see how to display HTML content in the notebook cell output

# COMMAND ----------

demographic_columns = ["Id","FirstName","LastName","Address","Email","Age"]
physicals_columns = ["Id","Pregnancies","BloodPressure","BMI"]
lab_result_columns = ["Id","Glucose","SkinThickness","Insulin", "DiabetesPedigreeFunction", "Outcome"]

# COMMAND ----------

#only for EDA
df.write.mode('overwrite').saveAsTable(raw_data_table)

# COMMAND ----------

df.select(demographic_columns).write.mode('overwrite').saveAsTable(demographic_table)

# COMMAND ----------

df.select(lab_result_columns).write.mode('overwrite').saveAsTable(lab_results_table)

# COMMAND ----------

df.select(physicals_columns).write.mode('overwrite').saveAsTable(physicals_results_table)
