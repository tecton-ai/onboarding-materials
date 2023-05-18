# Databricks notebook source
# MAGIC %md
# MAGIC # 📚 Feature Services, Versioning, Offline & Online Retrieval
# MAGIC
# MAGIC In this tutorial notebook, you will:
# MAGIC
# MAGIC 1. Retrieve data from the Feature Store via the SDK for model training
# MAGIC 2. Retrieve data from the Feature Store via the SDK for batch inference
# MAGIC 3. Retrieve data from the Feature Store API endpoint for real-time inference
# MAGIC 4. Explore payload requests and options to the API
# MAGIC 5. Create a variant (new version) of the API using a subset of feature views
# MAGIC 6. Test the variant

# COMMAND ----------

import tecton

import datetime
from datetime import datetime, date
from datetime import timedelta

# Check Tecton version
tecton.version.summary()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Select your workspace
# MAGIC
# MAGIC This tutorial will explicitly select the *prod* workspace below; you may want to select another for development created within a prior tutorial, based on development and materialization needs.

# COMMAND ----------

# list workspaces
tecton.list_workspaces()

# COMMAND ----------

# switch to workspace
ws = tecton.get_workspace('prod')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### What is a Feature Service?
# MAGIC
# MAGIC A Feature Service is a collection of features available in Tecton tied together for consumption.  Typically mapped 1:1 with a model.  They consist of 1 or more Feature Views.
# MAGIC Via the SDK, the service is used to create batch data sets as dataframes for batch inference or for point in time correct historic data sets for model training from the offline feature store.  
# MAGIC Each created feature service is also exposed via a low latency REST API to produce featre vectors in real time with retrievals from the online feature store.
# MAGIC
# MAGIC In this notebook, we'll be looking at the provided `fraud_detection_feature_service` in the trial tutorial.
# MAGIC
# MAGIC <img src ='https://www.doyouevendata.com/wp-content/uploads/2021/11/Screen-Shot-2021-11-04-at-2.23.55-PM.png'>
# MAGIC
# MAGIC Here we see and end to end view of data coming in, features being created and stored, and features ultimately tied together in the Feature Service where clients consume data via SDK or API.  This example includes Feature Views of multiple types; Batch, Streaming, and On Demand Feature Views are all included.
# MAGIC
# MAGIC
# MAGIC ### Constructing a training data set with historic point in time data retrieval
# MAGIC Features are retrieved given an entity id to look them up by, and a point in time.  Let's use the SDK to retrieve a historic data set which would be leveraged for model training.  Most features in our feature service are related to the `user` entity, however the 2 on on demand features operate based on the transaction `amount`.  We will need to provide Tecton a user to look up for retrieval, a timestamp to specify at which time in the past we'd like to know those features, and the amount for the on demand features to operate on.  These can be retrieved from the transactions_batch data source in the case of this tutorial.  You might imagine this as your transaction history in a data warehouse.  We will also pull our machine learning target, isfraud.

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
LIMIT 100
'''
training_events = spark.sql(transactions_query).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC In Tecton, the above data set is called a **spine**.  We will enrich it with historical features by passing it to the `get_historical_features` method for the `fraud_detection_feature_service`.  Note the (optional, default _False_) `from_source` parameter in the call.  Tecton could retrieve this data by going back to source and running the entire pipeline - however it will always be much faster to precompute them and materialize them into the offline feature store.  You can read more about this here: [Am I running GHF with materialized features?](https://docs.tecton.ai/docs/reading-feature-data/how-get-historical-features-works#materialized-or-non-materialized-features)

# COMMAND ----------

fraud_detection_feature_service = ws.get_feature_service("fraud_detection_feature_service")
training_data = fraud_detection_feature_service.get_historical_features(spine=training_events, timestamp_key="timestamp", from_source=False).to_spark()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constructing a batch data set with current features for batch inference
# MAGIC The process for retrieving a large data set for use in batch scoring through a model uses the same approach as above.  The difference is, rather than retrieving features for what the world looked like at a certain point in time to support model training - we simply pass in the timestamp of _right now_ to get the current features.  In this example, we'll just use the same spine as above but add a current_timestamp column; note that its value is now the one passed in to the `get_historical_features` parameter `timestamp_key`.

# COMMAND ----------

from pyspark.sql import functions as F

fraud_detection_feature_service = ws.get_feature_service("fraud_detection_feature_service")
scoring_data = fraud_detection_feature_service.get_historical_features(spine=training_events.withColumn("current_timestamp", F.current_timestamp()), 
  timestamp_key="current_timestamp", from_source=False).to_spark()
display(scoring_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Real time retrieval from the feature store via API
# MAGIC Tecton supports a very low latency real time retrieval API using the same Feature Service.  One of the benefits from using the same feature service for both training and scoring consumption is avoiding training/serving skew.  The API retrieves data from the online feature store, currently backed by either DynamoDB or Redis.  Crafting an HTTP POST requrest requires a TECTON_API_KEY - assumed to already be created from the base tutorial.  A new key can be generated from the Tection CLI if required.  
# MAGIC <pre>
# MAGIC $ tecton create-api-key
# MAGIC </pre>
# MAGIC
# MAGIC The Tecton instance is also required to run the python POST request example below.
# MAGIC
# MAGIC #### Feature Server Capabilities and SLAs
# MAGIC
# MAGIC Note that DynamoDB or Redis can be leveraged for the online feature store; DynamoDB is serverless; Redis on AWS is a service, but does require provisioning.  Provisioned resources are the responsibility of the customer for Redis.  The Feature Service is highly horizontally scalable and has been tested up to _hundreds of thousands_ of Queries Per Second (QPS) with a DynamoDB backend!  Check out our blog post on these benchmarks [here](https://www.tecton.ai/blog/serving-100000-feature-vectors-per-second-with-tecton-and-dynamodb/) - Feature Servers are provisioned on the Tecton side, so please work with us to advise whether your expectations are for 10 QPS or 100,000 QPS.  See the doc for more on [Service Level Objectives](https://docs.tecton.ai/v2/overviews/monitoring_feature_serving_slo.html).  The Tecton Feature Server does more than simply retrieve data from the store as well.  It is also where [On-Demand Feature Views](https://docs.tecton.ai/v2/examples/feature-definition-examples/on-demand-feature-view.html) are executed, operating on data only provided at request time.  Many feature aggregations are also calculated here, as Tecton cleverly performs them on tiles of stored data; see this blog post for more: [Real-Time Aggregation Features for Machine Learning](https://www.tecton.ai/blog/real-time-aggregation-features-for-machine-learning-part-1/).

# COMMAND ----------

TECTON_API_KEY = "1234567890cf633b4c5e2e234" #example: 1234567890cf633b4c5e2e234
TECTON_API_URL = "demo-applesauce.tecton.ai" #example: demo-applesauce.tecton.ai

# COMMAND ----------

# MAGIC %md
# MAGIC The HTTP request below constructed in python includes the optional `metadata_options`, with the `include_names` option toggled on.  See the [documentation](https://docs.tecton.ai/v2/examples/fetch-real-time-features.html#metadata-options-for-the-rest-api) for additional options.

# COMMAND ----------

import numpy as np
import pandas as pd
import requests
import json

url = 'https://%s/api/v1/feature-service/get-features' % TECTON_API_URL
headers = {'Authorization': 'Tecton-key ' + TECTON_API_KEY}

payload = """
{
  "params": {
    "feature_service_name": "fraud_detection_feature_service",
    "join_key_map": {
      "user_id": "user_461615966685",
      "merchant": "grocery_net"
    },
    "request_context_map": {
      "amt": 234.55
    },
    "workspace_name": "prod",
    "metadata_options": {
      "include_names": true
    }
  }
}
"""

response= requests.post(url, data=payload, headers=headers)

if response.status_code != 200:
  print("uh oh!  error: " + str(response.content))
else:
  print("successful response!")

# COMMAND ----------

# MAGIC %md
# MAGIC Depending on the model expectations, the retrieved data may be available to immediately feed the model, or additional manipulation may be needed.  A Python list is provided below; see the Appendix area for examples of manipulating the json feature vector response with python.
# MAGIC
# MAGIC ##### Python list

# COMMAND ----------

response.json()['result']['features']

# COMMAND ----------

# MAGIC %md
# MAGIC A request can be assembled and called via curl from the command line as follows:
# MAGIC <pre>
# MAGIC curl -X POST https://[YOUR-SERVER.tecton.ai]/api/v1/feature-service/get-features\
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
# MAGIC       "include_names": true
# MAGIC     }
# MAGIC   }
# MAGIC }'
# MAGIC </pre>
# MAGIC
# MAGIC Note the Tecton SDK itself can also be used to construct a request to the online feature store as well.

# COMMAND ----------

from pprint import pprint
fraud_detection_feature_service = ws.get_feature_service("fraud_detection_feature_service")

keys = {'user_id': 'user_461615966685', 'merchant': 'grocery_net'}
request_data = {'amt': 1122.14}

resp = fraud_detection_feature_service.get_online_features(keys, request_data=request_data)
pprint(resp.to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a variant of a Feature Service
# MAGIC When creating a new Feature Service, a completely new name can be defined; we can also create [a variant](https://docs.tecton.ai/docs/running-in-production/production-best-practices#use-variants-when-updating-feature-views-and-services) of a Feature Service of the same name that already exists.  This tutorial will take you through an example of culling the original service down to a subset as an example of some changes.  The changes will be the following:
# MAGIC
# MAGIC Start with the original `fraud_detection_feature_service`, but
# MAGIC 1. Remove the on demand feature `transaction_amount_is_higher_than_average` and incremental backfill batch feature `user_distinct_merchant_transaction_count_30d`.
# MAGIC 2. Currently the entire Feature View `user_transaction_amount_metrics` is brought in, which conists of means and sums across 4 windows apiece; 8 features.  Keep only the sum and mean for the 1 hour and 3 day windows.
# MAGIC 3. Name the variant `v3`
# MAGIC
# MAGIC Try doing this on your own before reading the cell below!

# COMMAND ----------

# MAGIC %md
# MAGIC Instructions to perform the above changes are as follows.
# MAGIC
# MAGIC Navigate to `fraud/feature_services/fraud_detection.py` and paste in the new function below:
# MAGIC <pre>
# MAGIC fraud_detection_feature_service_v3 = FeatureService(
# MAGIC     name='fraud_detection_feature_service:v3',
# MAGIC     description='A FeatureService providing features for a model that predicts if a transaction is fraudulent.',
# MAGIC     tags={'release': 'production'},
# MAGIC     features=[
# MAGIC         user_transaction_amount_metrics[['amt_mean_1h_10m', 'amt_mean_3d_10m', 'amt_sum_1h_10m', 'amt_sum_3d_10m']],
# MAGIC         user_transaction_counts,
# MAGIC         merchant_fraud_rate
# MAGIC     ]
# MAGIC )
# MAGIC </pre>
# MAGIC
# MAGIC Go to the command line and run `tecton apply` to push these changes to the Tecton server.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the variant Feature Service
# MAGIC Note that the original feature service required a user_id for point in time lookups of all stored features; however amount was required for the on demand features.  Since it is no longer used, we can drop it from the spine or POST API requests.  The new spine only requires a user_id and a point in time to look up.

# COMMAND ----------

display(training_events.drop("amt"))

# COMMAND ----------

# MAGIC %md
# MAGIC Requests to use the new variant Feature Service must include the full name as defined in the Feature Service declaration; those the one retrieved here or used in the API would be `fraud_detection_feature_service:v3`.

# COMMAND ----------

fraud_detection_feature_service = ws.get_feature_service("fraud_detection_feature_service:v3")
training_data = fraud_detection_feature_service.get_historical_features(spine=training_events.drop("amt"), timestamp_key="timestamp").to_spark()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Above we can see the successful retrieval of a new training data set, constructed with the subset of features originally defined in the base `fraud_detection_feature_service`.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Appendix: Python Feature Vector API Response Manipulation
# MAGIC
# MAGIC The following examples include various ways one might manipulate the json response payload returned by the API for model consumption.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Headerless CSV

# COMMAND ----------

",".join(str(fld) for fld in response.json()['result']['features'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Alternate Headerless CSV

# COMMAND ----------

pd.DataFrame([response.json()['result']['features']], columns=[y['name'] for y in response.json()['metadata']['features']]).to_csv(index=None, header=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### CSV w/Header

# COMMAND ----------

",".join(str(fld['name']) for fld in response.json()['metadata']['features']) + '\n' + ",".join(str(fld) for fld in response.json()['result']['features'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Alternate CSV w/Header

# COMMAND ----------

pd.DataFrame([response.json()['result']['features']], columns=[y['name'] for y in response.json()['metadata']['features']]).to_csv(index=None, header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Records oriented JSON

# COMMAND ----------

pd.DataFrame([response.json()['result']['features']], columns=[y['name'] for y in response.json()['metadata']['features']]).to_json(orient="records")

# COMMAND ----------

