# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Train and Serve a Model with Tecton + Databricks
# MAGIC 
# MAGIC Tecton is built to make it easy to build new ML models and move them production.  In this tutorial we'll see how you can:
# MAGIC * Generate Training Data with Tecton
# MAGIC * Train a model with that data, and track your experiments with MLFlow
# MAGIC * Register your trained model with Databricks's model registry
# MAGIC * Deploy your trained model to an endpoint with Databricks
# MAGIC * Perform inference on your deployed model, getting features from Tecton in real-time
# MAGIC 
# MAGIC 
# MAGIC **This tutorial should take you 30 minutes to complete**

# COMMAND ----------

# MAGIC %md
# MAGIC First, we'll make sure MLFlow is installed on this Spark cluster.

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Then some imports...

# COMMAND ----------

import mlflow
import mlflow.sklearn
import os
import requests
import tecton

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step One: Choose what features you want to use for your model
# MAGIC The first step of generating training data is to specify which features you'll need.  This is done in Tecton with a concept called a [Feature Service](https://docs.tecton.ai/v2/overviews/framework/feature_services.html), which provides a tool to logically group together the features needed for a model.  In this example, we'll use [this fraud detection feature service:](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/feature_services/fraud_detection.py).  You can learn more about generating training data in the tutorial named **Generating Training Data**.

# COMMAND ----------

ws = tecton.get_workspace('david-0-4-0-dogfood')
feature_service_name = "fraud_detection_feature_service"
fraud_detection_feature_service = ws.get_feature_service(feature_service_name)
fraud_detection_feature_service.summary()

# COMMAND ----------

# 1. Fetching a Spark DataFrame of historical labeled transactions
# 2. Renaming columns to match the expected join keys for the Feature Service
# 3. Selecting the join keys, request data, event timestamp, and label
training_events = ws.get_data_source("transactions_batch").get_dataframe().to_spark() \
                        .filter("partition_0 == 2022").filter("partition_2 == 05") \
                        .select("user_id", "merchant", "timestamp", "amt", "is_fraud") \
                        .cache()
display(training_events)

# COMMAND ----------

training_data = fraud_detection_feature_service.get_historical_features(spine=training_events, timestamp_key="timestamp").to_spark().cache()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step Two: Training an ML Model
# MAGIC 
# MAGIC Now that we've generated training data with Tecton, we need to prepare the data for modeling.  In this case our data is pretty small, so we'll move our data to a Pandas DataFrame, and prepare it for training.
# MAGIC 
# MAGIC ## Move to Pandas

# COMMAND ----------

training_data_pd = training_data.drop("user_id", "merchant", "timestamp", "amt").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Split features and labels

# COMMAND ----------

y = training_data_pd['is_fraud']
x = training_data_pd.drop('is_fraud', axis=1).fillna(0)
x

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split into training and test sets

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training a model + Using MLFlow
# MAGIC 
# MAGIC Next, lets train a simple [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) on our training data with sklearn.  We'll also use MLFlow to track our experiment and store the artifacts (like the trained model).
# MAGIC 
# MAGIC Note that we also log the `tecton_feature_service`, meaning our tracked experiment will know what feature set was used for training!

# COMMAND ----------

with mlflow.start_run() as run:
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  # Make predictions
  predictions = rf.predict(X_test)
  
  # Log parameters
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  mlflow.log_param("tecton_feature_service", feature_service_name)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
    
  # Log metrics
  mlflow.log_metric("mse", mse)


# COMMAND ----------

# MAGIC %md
# MAGIC #Register Model with Databricks and Create Serving Endpoint 
# MAGIC 
# MAGIC Databricks also hosts an MLFlow model registry that you can use to register models that are ready for production.  We're going to follow [these instructions from the Databricks Documentation](https://docs.databricks.com/applications/mlflow/model-registry-example.html#register-and-manage-the-model-using-the-mlflow-ui) to register a model.
# MAGIC 
# MAGIC ### Navigate to the most recent "Run" and click the Register Model Button
# MAGIC <img src='https://docs.databricks.com/_images/mlflow_ui_register_model.png' width="40%"/>
# MAGIC 
# MAGIC ### Name the model "tecton-fraud-model"
# MAGIC <img src='https://docs.databricks.com/_images/register_model_confirm.png' width="40%"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Serving Endpoint
# MAGIC 
# MAGIC Next, [lets follow these docs](https://docs.databricks.com/applications/mlflow/model-serving.html) to create a serving endpoint in Databricks 
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/enable-serving.png' width='50%' />
# MAGIC 
# MAGIC <img src='https://docs.databricks.com/_images/enable-serving.gif' width='50%' />
# MAGIC 
# MAGIC Once the endpoint is read make note of the "Model URL" -- you'll need to paste it below:

# COMMAND ----------

model_url = 'https://tecton-production.cloud.databricks.com/model/0-4-0-dogfooding/1/invocations'
my_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC # Online inference with features from Tecton's online store

# COMMAND ----------


def score_model(dataset):
  headers = {'Authorization': f'Bearer {my_token}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

amount=12345.0
df = fraud_detection_feature_service.get_online_features(
  join_keys={'user_id': 'user_131340471060', 'merchant': 'fraud_Schmitt Inc'},
  request_data={"amt": amount}
).to_pandas().fillna(0)

prediction = score_model(df)

print(prediction[0])
