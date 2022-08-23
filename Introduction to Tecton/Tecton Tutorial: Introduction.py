# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction to Tecton

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import some packages
# MAGIC 
# MAGIC ✅ Run the cell below to import the packages you'll need for this tutorial

# COMMAND ----------

# Import Tecton and other libraries
import os
import tecton
import pandas as pd
from datetime import datetime, timedelta


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Interacting with Tecton
# MAGIC Your Tecton account has been seeded with data and some example features that you can use to test out Tecton.
# MAGIC 
# MAGIC First, you can check out some of the raw data that has been connected to Tecton -- historical transactions.  You'll notice we first select the [Tecton workspace](https://docs.tecton.ai/overviews/workspaces.html) that contains the objects we want to fetch.

# COMMAND ----------

# Check out the data source in Snowflake
ws = tecton.get_workspace('prod')
ds = ws.get_data_source('transactions_stream')
ds.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1) Preview the raw data directly

# COMMAND ----------

display(ds.get_dataframe().to_spark().limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2) Tecton Feature Views
# MAGIC 
# MAGIC In Tecton, features are registered as [Feature Views](https://docs.tecton.ai/overviews/framework/feature_views/feature_views.html).  These views contain all of the information needed to transform raw data (like transactions) into features (like what % of historical transactions at a given merchant ended up being fraudulant)?
# MAGIC 
# MAGIC Feature Views can make feature data available in two places:
# MAGIC * Offline: You can retrieve historical feature values using [time travel](https://www.tecton.ai/blog/time-travel-in-ml/)
# MAGIC * Online: You can retrieve current feature values in real time via Tecton's [real-time serving API](https://docs.tecton.ai/examples/fetch-real-time-features.html)
# MAGIC 
# MAGIC Feature Views can also be run ad-hoc for testing or previewing data using `.run()`. Let's run the "Merchant Fraud Rate" Feature View to view feature data from the last 30 days (sorted by the merchants with the highest fraud rate):

# COMMAND ----------

fv = ws.get_feature_view('merchant_fraud_rate')

start_time = datetime.utcnow()-timedelta(days=14)
end_time = datetime.utcnow()

features = fv.run(start_time=start_time, end_time=end_time).to_spark()
display(features.orderBy("is_fraud_mean_1d_1d", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Generating Training Data
# MAGIC Once you've built a number of features, you'll want to join them together to generate training data. 
# MAGIC 
# MAGIC ### 3.1) Tecton Feature Services
# MAGIC In Tecton, features that are needed for training or predictions are grouped together into a [Feature Service](https://docs.tecton.ai/overviews/framework/feature_services.html). Typically you have one FeatureService per ML model. Let's check out a Feature Service that we've already built.

# COMMAND ----------

fs = ws.get_feature_service('fraud_detection_feature_service')
fs.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC The `fraud_detection_feature_service` is comprised of 13 features that are meant to be used together to train a fraud detection model.
# MAGIC 
# MAGIC ### 3.2) Building a Spine
# MAGIC 
# MAGIC Let's use the `fraud_detection_feature_service` to train a model that scores transactions as either "Fraudulent" or "Non-Fraudulent".  To start, lets look up some labeled transactions that we'll use for training.
# MAGIC 
# MAGIC We can see in the summary above that the `fraud_detection_feature_service` requires `merchant` and `user_id` join keys in order to fetch all the relevant features. Together with an event timestamp and label column, this represents our list of historical training events. In Tecton we call this a "spine".
# MAGIC 
# MAGIC > 💡 A spine is expected to include the entity join keys for the Feature Views in a Feature Service as well as a timestamp column for time-travel lookups. A label column is not strictly necessary but is typically included if you generate a training dataset.

# COMMAND ----------

# Preview the data directly
yesterday = datetime.utcnow() - timedelta(days=1)

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
ORDER BY TIMESTAMP DESC
LIMIT 10000
'''
transactions = spark.sql(transactions_query).cache()
display(transactions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3) Getting Training Data with `get_historical_features`
# MAGIC 
# MAGIC To retrieve training data, we'll use Tecton's `get_historical_features` API, which allows us to join the 13 features contained in `fraud_detection_feature_service` onto our historical transactions.
# MAGIC 
# MAGIC 
# MAGIC A Feature Service will expect a spine in the form of a Spark or Pandas Dataframe.

# COMMAND ----------

training_data = fs.get_historical_features(spine=transactions, timestamp_key="timestamp").to_spark()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is happening behind the scenes
# MAGIC 
# MAGIC Behind the scenes, Tecton is doing a row-level, [point-in-time correct](https://www.tecton.ai/blog/time-travel-in-ml/) join.  This join logic helps you ensure that the data you use to train your models is drawn from the same distribution as the data that is likely to be used at production time.
# MAGIC 
# MAGIC One other helpful thing -- you never need to worry about different concepts of time in your data when generating training data. For each feature you can specify the most convenient or correct time for that feature, and Tecton's join logic will make it easy to join all of your features together.
# MAGIC 
# MAGIC <img src="https://docs.tecton.ai/v2/assets/docs/examples/point-in-time-correct-joins.png" width="50%" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Getting Real-Time Features for Inference
# MAGIC 
# MAGIC ### 4.1) Generate an API token
# MAGIC 
# MAGIC To fetch real-time (online) features at low latency for a production application we will use Tecton's REST API.
# MAGIC 
# MAGIC This will require creating an API key. In your terminal, run:
# MAGIC 
# MAGIC ✅ `$ tecton api-key create`
# MAGIC 
# MAGIC Then set this API key as an enviornment variable using the line below and replacing "<key>" with the generate API key:
# MAGIC 
# MAGIC ✅ `$ export TECTON_API_KEY=<key>`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2) Retrieve online features using the Python SDK
# MAGIC 
# MAGIC We can hit Tecton's REST API directly from the Python SDK using `fs.get_online_features(keys)`. This is convenient for testing purposes.
# MAGIC 
# MAGIC ✅ To query the REST API from the Python SDK, we need to set the API key in the first line of the cell below. Replace "\<key>" with the token generated in the step above.

# COMMAND ----------

from pprint import pprint

keys = {
    # 'MERCHANT': 'fraud_Gutmann Ltd',
    'user_id': 'user_461615966685',
    'merchant': 'grocery_net'
}

request_data = {
  'amt': 150.
}
features = fs.get_online_features(join_keys=keys, request_data=request_data).to_dict()
pprint(features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3) Retrieve features directly from the REST API via a cURL
# MAGIC 
# MAGIC We can also directly query Tecton's REST API using the example cURL below.
# MAGIC 
# MAGIC ✅ Run this in your terminal, but make sure to replace `<your-cluster>` cluster name in the first line with your cluster name:
# MAGIC 
# MAGIC ```bash
# MAGIC curl -X POST --silent https://<your-cluster>.tecton.ai/api/v1/feature-service/get-features\
# MAGIC      -H "Authorization: Tecton-key $TECTON_API_KEY" -d\
# MAGIC '{
# MAGIC   "params": {
# MAGIC     "feature_service_name": "fraud_detection_feature_service",
# MAGIC     "join_key_map": {
# MAGIC       "USER_ID": "user_461615966685",
# MAGIC       "CATEGORY": "grocery_net"
# MAGIC     },
# MAGIC     "workspace_name": "prod"
# MAGIC   }
# MAGIC }' | jq
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # What's Next
# MAGIC 
# MAGIC Tecton is a powerful tool to build, manage, share, and consume features for ML.  Check out the next tutorial "Creating Features on Snowflake" to learn how to build your own features.
