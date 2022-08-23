# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Using Tecton to Generate Training Data
# MAGIC 
# MAGIC Once you've built all the features you need for an ML model, Tecton makes it simple to generate training data. This tutorial will walk through:
# MAGIC 
# MAGIC * How to build a traning dataset with Tecton
# MAGIC * What tools Tecton provides to help with training / serving skew
# MAGIC 
# MAGIC <img src="https://docs.tecton.ai/v2/assets/docs/examples/generate-training-data-overview.png" width='50%'/>
# MAGIC 
# MAGIC 
# MAGIC **This tutorial should take you 30 minutes to complete**

# COMMAND ----------

# Setup
import tecton
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step One: Choose what features you want to use for your model
# MAGIC 
# MAGIC The first step of generating training data is to specify which features you'll need.  This is done in Tecton with a concept called a [Feature Service](https://docs.tecton.ai/v2/overviews/framework/feature_services.html), which provides a tool to logically group together the features needed for a model.  In this example, we'll use [this fraud detection feature service:](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/feature_services/fraud_detection.py)
# MAGIC 
# MAGIC ```python
# MAGIC fraud_detection_feature_service = FeatureService(
# MAGIC     name='fraud_detection_feature_service',
# MAGIC     features=[
# MAGIC         transaction_amount_is_higher_than_average,
# MAGIC         user_transaction_amount_metrics,
# MAGIC         user_transaction_counts,
# MAGIC         user_distinct_merchant_transaction_count_30d,
# MAGIC         merchant_fraud_rate
# MAGIC     ]
# MAGIC )
# MAGIC ```
# MAGIC The key component of a feature service is simple -- a list of the feature views you'd like to include.  Here you can see we've included five feature views, which means we should expect to retreive all of the features within these five feature views when we fetch training data.

# COMMAND ----------

ws = tecton.get_workspace('david-0-4-0-dogfood')
fs = ws.get_feature_service('fraud_detection_feature_service')
fs.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Two: Specify the training rows you want to retreive
# MAGIC 
# MAGIC Now that you have selected the features you want to retreive, you'll need to specify what rows of data you'd like. You do this by creating a **spine dataframe** -- a dataframe that contains **keys** and **timestamps** which Tecton will use to index into the Feature Store. Most often, the rows you'll be looking up correspond to *historical prediction events* -- points in the past where your model would have made a prediction.
# MAGIC 
# MAGIC For this example, we'll be building training data for a **fraud detection model** that predicts whether or not a transaction is fraudulent at the time of the transaction. To train this model, we'll look up historical transactions that are classified as "Payments" which are labeled as either fradulent or non-fradulent.  Those historical transactions will become our **spine dataframe**. Since this feature service also contains an [On-Demand Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/on_demand_feature_view.html) that depends on the `amount` of a transaction, we'll also surface the amount of the historical transaction.

# COMMAND ----------

from pyspark.sql import functions as F

# 1. Fetching a Spark DataFrame of historical labeled transactions
# 2. Renaming columns to match the expected join keys for the Feature Service
# 3. Selecting the join keys, request data, event timestamp, and label
spine = ws.get_data_source("transactions_batch").get_dataframe().to_spark() \
                        .select("user_id", "merchant", "amt", "timestamp", "is_fraud") \
                        .filter("partition_0 == '2022'") \
                        .filter("partition_1 == '05'") \
                        .limit(1000).cache()
display(spine)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Three: Retreive training data
# MAGIC 
# MAGIC Once we have our spine dataframe, the rest is easy -- we'll use the `get_historical_features` method to look up the features for each of these rows.

# COMMAND ----------

training_data = fs.get_historical_features(spine)

# COMMAND ----------

display(training_data.to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is happening behind the scenes
# MAGIC 
# MAGIC Behind the scenes, Tecton is doing a row-level, [point-in-time correct](https://www.tecton.ai/blog/time-travel-in-ml/) join.  This join logic helps you ensure that the data you use to train your models is drawn from the same distribution as the data that is likely to be used at production time.
# MAGIC 
# MAGIC One other helpful thing -- you never need to worry about different concepts of time in your data when generating training data. For each feature you can speicify the most convenient or correct time for that feature, and Tecton's join logic will make it easy to join all of your features together.
# MAGIC 
# MAGIC <img src="https://docs.tecton.ai/v2/assets/docs/examples/point-in-time-correct-joins.png" width="50%" />
