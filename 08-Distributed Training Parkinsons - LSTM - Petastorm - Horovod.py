# Databricks notebook source
# MAGIC %md
# MAGIC #### Read data

# COMMAND ----------

# MAGIC %md
# MAGIC **My GPU Cluster Settings**
# MAGIC ```
# MAGIC 8 Workers,128 GB Memory,32 Cores
# MAGIC 1 Driver,16 GB Memory, 4 Cores
# MAGIC Runtime
# MAGIC 13.3LTS ML GPU
# MAGIC ```
# MAGIC
# MAGIC **My Non GPU Cluster Settings**
# MAGIC ```
# MAGIC 8 Workers,128 GB Memory,32 Cores
# MAGIC 1 Driver,16 GB Memory, 4 Cores
# MAGIC Runtime
# MAGIC 13.3LTS ML
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

input_data_file = f"{raw_data_path}/Postural_Tremor_DA_Raw.csv"

# COMMAND ----------

raw_data = spark.read.csv(input_data_file, header=True, inferSchema=True)
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analyze Data
# MAGIC
# MAGIC First we need to understand how data is structured and how we can obtain timeseries
# MAGIC
# MAGIC Let us see if Patient Type(PATTYPE) repeats within a block. We need to validate our assumption that each block corresponds to the same patient type

# COMMAND ----------


from pyspark.sql.functions import count_distinct,col
an_df = raw_data.select("block","PATTYPE").groupBy("block").agg(count_distinct("PATTYPE").alias("d")).filter( col("d") > 1)

display(an_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The above query did not return any result. That means, the PATTYPE does not change within a block.
# MAGIC
# MAGIC This means that we can data for each block and use the same label for that group to train the model. There is no need to shift the label (as we normally do in time series analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC Let us check the data distribution of PATTYPE, just to understand how much of the data contain measurements for each PATTYPE

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import count,col,lit,floor,spark_partition_id, row_number, collect_list, struct,collect_set, explode, array,flatten

# COMMAND ----------

display(raw_data.select("block","time_ms").groupBy("block").agg(count("time_ms")))

# COMMAND ----------

display(raw_data.select("PATTYPE"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that there is a good representation of both PATTYPE

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare Data
# MAGIC Chunk the data into windows of timeseries data. For this example we will feed 50 time steps to obtain an output. This [documentation](https://www.tensorflow.org/tutorials/structured_data/time_series#1_indexes_and_offsets) by Tensorflow explain the concept of Windows for timeseries elaborately

# COMMAND ----------

from ast import literal_eval

feature_cols = ["user_acc_x_g","user_acc_y_g","user_acc_z_g","rot_x_rad/s","rot_y_rad/s","rot_z_rad/s"]
label_col = "PATTYPE"
time_steps = 10
label_width = 1

windowSpec = (Window
            .partitionBy("block")
            .orderBy(
                col("EDC_ID").asc(), 
                col("session_code").asc(), 
                col("block").asc(),
                col("time_ms").asc()
              )
            )

#We will group the data for same block into multiple windows. Before windowing we will make sure
#the data is in right order by sorting by time_ms
input_data = (raw_data      
      .withColumn("row_num",row_number().over(windowSpec) -1)      
      .withColumn("seq", floor(col("row_num")/lit(time_steps)))
      .select(["block","seq"] + feature_cols + [label_col])
      .groupBy("block","seq")
      .agg(
        collect_list(array(feature_cols)).alias("series"),
        collect_set(label_col).alias("label")
      )
      #Petastorm does not like 2d arrays..so we need to reshape later
      .withColumn("series", flatten(col("series")))
      .withColumn("label", explode(col("label")))
      .select("series","label")
     )


# COMMAND ----------

display(input_data)

# COMMAND ----------

#Now we have data grouped into windows, we can create train and test data
train_df, test_df = input_data.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Convert the Spark DataFrame to a TensorFlow Dataset
# MAGIC
# MAGIC In order to convert Spark DataFrames to a TensorFlow datasets, we will use [Petastorm](https://github.com/uber/petastorm) and we need to do it in two steps:
# MAGIC
# MAGIC
# MAGIC Define where you want to copy the data by setting Spark config
# MAGIC Call `make_spark_converter()` method to make the conversion
# MAGIC This will copy the data to the specified path.

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

if 'converter_train' in globals():
  converter_train.delete()
  
if 'converter_test' in globals():
  converter_test.delete()

#Databricks Utilities (dbutils): https://docs.databricks.com/en/dev-tools/databricks-utils.html
path_to_cache = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_cache"

dbutils.fs.rm(path_to_cache, recurse=True)
dbutils.fs.mkdirs(path_to_cache)

#Convert the spark df to tensorflow dataset
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, path_to_cache)
converter_train = make_spark_converter(train_df)
converter_test = make_spark_converter(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model
# MAGIC Let us create a simple Keras LSTM model

# COMMAND ----------

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.activation import LeakyReLU

import mlflow
import mlflow.keras

batch_size = 50
initial_lr = 0.1
num_epoch = 10
num_features = len(feature_cols) 

def build_model():
  model = Sequential()
  model.add(LSTM(units=20, input_shape=(time_steps,num_features) ))
  model.add(LeakyReLU(alpha=0.5)) 
  model.add(Dropout(0.1))
  model.add(Dense(1, activation='sigmoid'))
  return model
  


# COMMAND ----------

# MAGIC %md
# MAGIC Let us examine if the series is being reshaped properly

# COMMAND ----------

import tensorflow as tf
with converter_train.make_tf_dataset(batch_size=batch_size) as train_dataset:
  dataset = train_dataset.map( lambda x: (tf.reshape(x.series, [-1,time_steps,num_features]), x.label) )
  for inputs in dataset.take(1):
    print(inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good.. let us train the model

# COMMAND ----------

with converter_train.make_tf_dataset(batch_size=batch_size) as train_dataset:
    dataset = train_dataset.map( lambda x: (tf.reshape(x.series, [-1,time_steps,num_features]), x.label) )

    # Number of steps required to go through one epoch
    steps_per_epoch = len(converter_train) // batch_size
    
    model = build_model()
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=num_epoch, verbose=2)
    


# COMMAND ----------

 with converter_test.make_tf_dataset(num_epochs=1) as test_dataset:
  dataset = test_dataset.map( lambda x: (tf.reshape(x.series, [-1,time_steps,num_features]), x.label) )
  hist = model.evaluate(dataset)
  accuracy = hist[1]
  print(f"Accuracy: {accuracy}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parallelize using Horovod

# COMMAND ----------

# MAGIC %md
# MAGIC Petastorm writes its output to Parquet files. Each Parquet file consists of one or more internal row groups. Horovod expects the total number of row groups available to it to be equal to or greater than the number of shards (parallel processes) it employs. 
# MAGIC
# MAGIC To do this, we sum the bytes associated with each for each of the training and testing datasets and divide that number by the number of virtual CPUs (as presented by sc.defaultParallelism) to ensure we align with Horovod's requirements

# COMMAND ----------

from pyspark.sql.functions import size, sum

if 'converter_train' in globals():
  converter_train.delete()
  
if 'converter_test' in globals():
  converter_test.delete()
  
#Databricks Utilities (dbutils): https://docs.databricks.com/en/dev-tools/databricks-utils.html
path_to_cache = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_cache_dist"
dbutils.fs.rm(path_to_cache, recurse=True)
dbutils.fs.mkdirs(path_to_cache)

# determine rough bytes in dataset
bytes_in_train = (train_df
                  .withColumn('bytes', (size( col('series')) + lit(1)) ) 
                  .groupBy()
                  .agg(sum( col('bytes')) .alias('bytes'))
                  .collect()[0]['bytes'])

bytes_in_test = (test_df
                  .withColumn('bytes', (size( col('series')) + lit(1)) ) 
                  .groupBy()
                  .agg(sum( col('bytes')) .alias('bytes'))
                  .collect()[0]['bytes'])

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, path_to_cache)

converter_train = make_spark_converter(train_df, parquet_row_group_size_bytes=int(bytes_in_train/spark.sparkContext.defaultParallelism))
converter_test = make_spark_converter(test_df, parquet_row_group_size_bytes=int(bytes_in_test/spark.sparkContext.defaultParallelism))

# COMMAND ----------

### BEFORE RUNNING THIS CMD RUN THE PREVIOUS COMMAND AND RE INITIALIZE THE CONVERTER
import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner
import os
from datetime import datetime

checkpoint_dir = f"file:///dbfs/workshop/temp/cache/{user_name}/lstm/parkinsons/petastorm_checkpoint_weights"

dbutils.fs.rm(checkpoint_dir, True)
batch_size = 100
initial_lr = 0.1
num_epoch = 1

#Databricks Utilities (dbutils): https://docs.databricks.com/en/dev-tools/databricks-utils.html
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

#Create an MLFlow experiment
experiment_tag = f"{user_prefix}_parkinsons_prediction_{datetime.now().strftime('%d-%m-%Y')}"
experiment_base_path = f"Users/{user_email}/mlflow_experiments"
dbutils.fs.mkdirs(f"file:/Workspace/{experiment_base_path}")
experiment_path = f"/{experiment_base_path}/{experiment_tag}"

# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)
   
def run_training_horovod():
    # Horovod: initialize Horovod.
    import mlflow
    import mlflow.keras

    hvd.init()    
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    with converter_train.make_tf_dataset(batch_size=batch_size, 
                                         num_epochs=None, 
                                         cur_shard=hvd.rank(), 
                                         shard_count=hvd.size()) as train_dataset:
        
        dataset = train_dataset.map( lambda x: (tf.reshape(x.series, [-1,time_steps,num_features]), x.label) )
        
        model = build_model()
        steps_per_epoch = len(converter_train) // (batch_size*hvd.size())

        # Adding in Distributed Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        optimizer = hvd.DistributedOptimizer(optimizer)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        # Adding in callbacks
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=initial_lr*hvd.size(), warmup_epochs=5, verbose=2),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10, verbose=2)
        ]

        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, monitor="loss", save_best_only=True))

        print("Preparing to train the model")
        history = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=num_epoch, callbacks=callbacks, verbose=2)
        
        # MLflow Tracking (Log only from Worker 0)
        if hvd.rank() == 0:    

            # Log events to MLflow
            with mlflow.start_run(run_id = active_run_id) as run:
                # Log MLflow Parameters
                mlflow.log_param("num_layers", len(model.layers))
                mlflow.log_param("optimizer_name", "Adam")
                mlflow.log_param("learning_rate", initial_lr)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("hvd_np", hvd_np)

                # Log MLflow Metrics
                mlflow.log_metric("train loss", history.history["loss"][-1])

                # Log Model
                mlflow.keras.log_model(model, "model")

# COMMAND ----------


with mlflow.start_run() as run: 
    active_run_id = mlflow.active_run().info.run_id
    hvd_np = spark.sparkContext.defaultParallelism # change to num gpus when running on gpu cluster
    hr = HorovodRunner(np=hvd_np, driver_log_verbosity="all")
    hr.run(run_training_horovod)

# COMMAND ----------

test_df_pd = raw_data.select(feature_cols+[label_col]).toPandas()
test_df_pd

# COMMAND ----------

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, batch_size,
               train_df, val_df, test_df,
               feature_columns=None, label_columns=None):
    
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    
    self.batch_size = batch_size
    
    # Work out the label column indices.
    self.label_columns = label_columns
    self.feature_columns = feature_columns
    
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
    
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
    self.label_start = self.total_window_size - self.label_width    
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    if self.feature_columns is not None:
      inputs = tf.stack(
          [inputs[:, :, self.column_indices[name]] for name in self.feature_columns],
          axis=-1)
    
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size,)

    ds = ds.map(self.split_window)

    return ds

  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

# COMMAND ----------

from tensorflow.keras.models import load_model

trained_model = load_model(checkpoint_dir)
print(trained_model.summary())

wg = WindowGenerator(input_width=time_steps, 
                     label_width=label_width,
                     shift=1,
                     batch_size=batch_size,
                     train_df=test_df_pd,
                     val_df=test_df_pd,
                     test_df=test_df_pd,
                     feature_columns=feature_cols,
                     label_columns=[label_col])

trained_model.evaluate(wg.test)

# COMMAND ----------

if 'converter_train' in globals():
  converter_train.delete()
  
if 'converter_test' in globals():
  converter_test.delete()

# COMMAND ----------


