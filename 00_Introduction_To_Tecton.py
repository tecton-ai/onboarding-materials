# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction to Tecton
# MAGIC
# MAGIC In this tutorial we'll cover how you can use Tecton to work with some existing features.  We'll cover:
# MAGIC * How to view data from a Tecton data source
# MAGIC * How to view features from a Tecton feature view
# MAGIC * How to create a historic point in time correct training data set from a Tecton feature service
# MAGIC * How to retrieve a vector of data from the online store
# MAGIC
# MAGIC ### Import some packages
# MAGIC First we'll import the tecton package and some other libraries supporting this tutorial.

# COMMAND ----------

# import tecton and other libraries
import os
import tecton
import pandas as pd
from datetime import datetime, timedelta

# check tecton version
tecton.version.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interacting with Tecton
# MAGIC This Tecton account has been seeded with data and some example features that can be used to test out Tecton.
# MAGIC
# MAGIC First, we can take a look at some of the financial transaction data available in this Tecton environment.  This is the `transaction_stream` data source.  Notice we first select the [workspace](https://docs.tecton.ai/overviews/workspaces.html) that contains the objects we want to fetch, in this case named `prod`.  We'll also display some metadata about this data source below, then preview some of the data from the offline store.
# MAGIC
# MAGIC Where does this data live?  The `transaction_stream` data source is a streaming data source, defined by a streaming source (kinesis) that is used to fill the online store, and a batch source (partitioned parquet files on s3, indexed by the hive catalog supporting this compute backend) used to fill the offline store.  External to Tecton the streaming event data is written to S3 to efficiently support the processing of historical data and the offline store.

# COMMAND ----------

# check out a data source in the prod workspace
ws = tecton.get_workspace('prod')
ds = ws.get_data_source('transactions_stream')
ds.summary()

# COMMAND ----------

display(ds.get_dataframe().to_spark().limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.) Tecton Feature Views (FVs)
# MAGIC
# MAGIC In Tecton, features are registered as [Feature Views](https://docs.tecton.ai/overviews/framework/feature_views/feature_views.html).  These views contain all of the information needed to transform raw data (like transactions) into features (what percentage of historical transactions at a given merchant ended up being fraudulant?)
# MAGIC
# MAGIC Feature Views can make feature data available in two places:
# MAGIC * Offline: You can retrieve historical feature values using [time travel](https://www.tecton.ai/blog/time-travel-in-ml/), or create batch inference data sets
# MAGIC * Online: You can retrieve current feature values in real time via Tecton's [real-time serving API](https://docs.tecton.ai/examples/fetch-real-time-features.html)
# MAGIC
# MAGIC Feature Views can also be run ad-hoc for testing or previewing data using `.run()`. Let's run the "Merchant Fraud Rate" Feature View to view feature data from the last 30 days (sorted by the merchants with the highest single day fraud rate). This is being computed on demand with the cluster this notebook is running in.  Note the start time of 30 days prior only allows for 1d and 30d fraud values to be fully calculated below.  Change this to 90 days or larger to compute all features available in this view fully.  Also note the same merchant may show up on multiple lines, at different timestamps.  This is showing the fraud rate for a merchant as of that row's timestamp; there are multiple entries for a merchant due to this changing each day.

# COMMAND ----------

fv = ws.get_feature_view('merchant_fraud_rate')

start_time = datetime.utcnow().replace(microsecond=0, second=0, minute=0, hour=0)-timedelta(days=30)
end_time = datetime.utcnow().replace(microsecond=0, second=0, minute=0, hour=0)

features = fv.run(start_time=start_time, end_time=end_time).to_spark()
display(features.orderBy("is_fraud_mean_1d_1d", ascending=False).limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.) Generating Training Data with a Tecton Feature Service
# MAGIC Once features have been built in Tecton, they can be grouped together into a [Feature Service](https://docs.tecton.ai/overviews/framework/feature_services.html) for consumption. Typically each use case or ML model is associated with its own feature service.  Let's look at an existing feature service, the `fraud_detection_feature_service`.  Some metadata about what's in the feature service and how to use it can be provided via the SDK.  The `fraud_detection_feature_service` is comprised of 14 features that are meant to be used together for training and scoring a fraud detection model.

# COMMAND ----------

fs = ws.get_feature_service('fraud_detection_feature_service')
fs.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1) Building a Spine
# MAGIC
# MAGIC Let's use the `fraud_detection_feature_service` to train a model that scores transactions as either "Fraudulent" or "Non-Fraudulent" (1 or 0).  A spine in Tecton is a batch set of records that will be enriched with features retrieved from the offline feature store.  The spine for our training data set is past transactions with our labeled ML target (`is_fraud`) - the format of the spine consists of entity keys known for the transaction at the time, as well as the time it occurred.  Per the feature service metadata above, we can see the feature service requires as input a `user_id` and a `merchant` as entity keys; these are the keys that own all the features and how the features will be looked up.  The `amt` column is also expected and used, as an input to an [On-Demand Feature View](https://docs.tecton.ai/docs/defining-features/feature-views/on-demand-feature-view). This value will be featurized in real-time during the request. Additional fields - such as the target label - are not required for feature retrieval but will be echoed back in the result data set.

# COMMAND ----------

yesterday = datetime.utcnow() - timedelta(days=3)

# Data is partitioned by year, month, day
partition_year = str(yesterday.year)
partition_month = str(yesterday.month).zfill(2)
partition_day = str(yesterday.day).zfill(2)

transactions_query = f'''
SELECT 
    merchant,
    user_id,
    to_timestamp(timestamp) as timestamp,
    amt,
    is_fraud
FROM 
    demo_fraud_v2.transactions
WHERE partition_0 = '{partition_year}' and partition_1 = '{partition_month}' and partition_2 = '{partition_day}'
LIMIT 1000
'''
transactions = spark.sql(transactions_query).cache()
display(transactions.limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2) Getting Training Data with `get_historical_features`
# MAGIC
# MAGIC To retrieve training data, we'll use Tecton's `get_historical_features` API, which allows us to join the 14 features contained in `fraud_detection_feature_service` onto our historical transactions. A Feature Service will expect a spine in the form of a Spark or Pandas Dataframe. The timestamp_key specified is the column which we'll use for historic look up of features; note this is done on a record by record basis and does not require any kind of rounding or inner joins; time travel can be done to any timestamp value.  Also note times are in GMT.

# COMMAND ----------

training_data = fs.get_historical_features(spine=transactions, timestamp_key="timestamp").to_spark()
display(training_data.limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC #### What is happening behind the scenes
# MAGIC
# MAGIC Behind the scenes, Tecton is doing a row-level, [point-in-time correct](https://www.tecton.ai/blog/time-travel-in-ml/) join.  This join logic helps you ensure that the data you use to train your models is drawn from the same distribution as the data that is likely to be used at production time.
# MAGIC
# MAGIC One other helpful thing -- you never need to worry about different concepts of time in your data when generating training data. For each feature you can specify the most convenient or correct time for that feature, and Tecton's join logic will make it easy to join all of your features together.
# MAGIC
# MAGIC <br>
# MAGIC <center><img src="https://docs.tecton.ai/assets/images/point-in-time-correct-joins-b2a1b25e3fa5072bf71b21237caf7f81.png" width="50%" /></center>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.) Retrieving the latest features from the online store
# MAGIC Once a model has been trained and deployed, the pipelines are already in place to feed it features in batch or real-time.  Real-time features are retrieved from the online store via a highly scalable low latency API.
# MAGIC
# MAGIC #### 3.1) Using the SDK to retrieve online features
# MAGIC We can hit Tecton's REST API directly from the Python SDK using `fs.get_online_features(keys)`. This is convenient for testing purposes, however adds some metadata overhead so this method is not advised for production retrievals.

# COMMAND ----------

from pprint import pprint

# entity keys required to look up features owned by user_id and merchant in the online store
keys = {
    'user_id': 'user_461615966685',
    'merchant': 'grocery_net'
}

# real-time data provided at the time of request as input to on-demand feature values
request_data = {
  'amt': 150.34
}
features = fs.get_online_features(join_keys=keys, request_data=request_data).to_dict()
pprint(features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2) Using the API to retrieve online features - install the Tecton CLI on your local machine
# MAGIC Follow these instructions in the Tecton Docs to set up the [Tecton CLI](https://docs.tecton.ai/docs/setting-up-tecton/development-setup/installing-the-tecton-cli) -- this will be used for several operations when working with Tecton, as well as within subsequent tutorials.
# MAGIC
# MAGIC `pip install 'tecton[pyspark]'`
# MAGIC #### 3.3) Log in to the Tecton instance and create an API key
# MAGIC `tecton login` # fill in the tecton instance from the prompt to initiate a browser based login, eg. it will look similar to: https://demo-peach.tecton.ai
# MAGIC
# MAGIC Next we'll create an API key to use the REST API.
# MAGIC
# MAGIC `tecton api-key create`
# MAGIC
# MAGIC Tke the resultant API key output and export it as an environment variable.
# MAGIC
# MAGIC `export TECTON_API_KEY=<key>`
# MAGIC
# MAGIC This can be done in your local shell environment and subsequently tested with a curl request to the API.  It can similarly be tested below in the notebook. __Replace__ the tecton instance name (demo-peach) with your own Tecton instance.

# COMMAND ----------

import os
os.environ['TECTON_API_KEY'] = "123456bac6a66b1cc1cf633b4c5e2e234"

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -X POST --silent https://demo-peach.tecton.ai/api/v1/feature-service/get-features\
# MAGIC      -H "Authorization: Tecton-key $TECTON_API_KEY" -d\
# MAGIC '{
# MAGIC   "params": {
# MAGIC     "feature_service_name": "fraud_detection_feature_service",
# MAGIC     "join_key_map": {
# MAGIC       "user_id": "user_461615966685",
# MAGIC       "merchant": "grocery_net"
# MAGIC     },
# MAGIC     "request_context_map": {
# MAGIC       "amt": 234.55
# MAGIC     },
# MAGIC     "workspace_name": "prod"
# MAGIC   }
# MAGIC }' 

# COMMAND ----------

# MAGIC %md
# MAGIC Additional metatdata options can be specified as well, such as responding with the feature names and effective times of the features.

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -X POST --silent https://demo-peach.tecton.ai/api/v1/feature-service/get-features\
# MAGIC      -H "Authorization: Tecton-key $TECTON_API_KEY" -d\
# MAGIC '{
# MAGIC   "params": {
# MAGIC     "feature_service_name": "fraud_detection_feature_service",
# MAGIC     "join_key_map": {
# MAGIC       "user_id": "user_461615966685",
# MAGIC       "merchant": "grocery_net"
# MAGIC     },
# MAGIC     "request_context_map": {
# MAGIC       "amt": 234.55
# MAGIC     },
# MAGIC     "workspace_name": "prod",
# MAGIC     "metadata_options": {
# MAGIC       "include_names": true,
# MAGIC       "include_effective_times": true
# MAGIC     }
# MAGIC   }
# MAGIC }' 

# COMMAND ----------

