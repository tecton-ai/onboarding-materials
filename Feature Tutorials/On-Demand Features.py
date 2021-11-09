# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #On-Demand Feature Tutorial
# MAGIC 
# MAGIC Tecton has 5 basic types of Feature Views:
# MAGIC - [Batch Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_feature_view.html)
# MAGIC - [Batch Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_window_aggregate_feature_view.html)
# MAGIC - [Stream Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/stream_feature_view.html)
# MAGIC - [Stream Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/stream_window_aggregate_feature_view.html)
# MAGIC - [On-Demand Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/on_demand_feature_view.html)
# MAGIC 
# MAGIC In this tutorial we'll focus on **On-Demand Feature Views**
# MAGIC 
# MAGIC 
# MAGIC **This tutorial should take about 30 minutes to complete**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## What is an On-Demand Feature
# MAGIC 
# MAGIC Most of the features that you'll build in Tecton are **precomputed** -- this means that Tecton will run the data pipelines needed to compute these features before they are needed, and your ML applications will simply look up precomputed feature values from Tecton.
# MAGIC 
# MAGIC In some scenarios, the model of precomputing features doesn't make sense, and instead you'd rather compute the value of a feature **on-demand**.  Some examples:
# MAGIC * You need access to data that is only available just before you need to make a prediction
# MAGIC   * (example) a user just made a transaction, and you want to compute features about that transaction
# MAGIC   * (example) a user just filled out a form in your application, and you want to featurize the data they entered
# MAGIC * Precomputing features is inefficient because most of the features will never be used
# MAGIC   * (example) you want to calculate two users mutual friends, but precomputing mutual friends for every user is infeasible
# MAGIC   
# MAGIC   
# MAGIC For these scenarios, Tecton has support for **On-Demand Features** -- features that are dynamically computed when requesting features for inference. 
# MAGIC 
# MAGIC ## How do they work
# MAGIC 
# MAGIC ### Writing On-Demand Features
# MAGIC On-Demand Features are written in declaritive code just like all other features in Tecton, however there is one key difference: **on-demand feature views are written in Python (using Pandas primitives)**.  This tutorial will guide you through how they work.
# MAGIC 
# MAGIC ### At Inference Time
# MAGIC At inference time, the transformation logic for on-demand feature are run directly on the Tecton-managed serving infrastructure. Tecton has developed an efficient method to quickly invoke python functions at serving time without inducing significant overhead. How this works:
# MAGIC 
# MAGIC 1. When you invoke [Tecton's Feature Serving API](https://docs.tecton.ai/v2/examples/fetch-real-time-features.html), you'll include any request-time data that needs to be processed in one-or-more on-demand features.
# MAGIC 2. While Tecton is looking up any precomputed features, Tecton will also invoke your on-demand transformation logic to compute the on-demand feature on the fly.
# MAGIC 3. Tecton will return a feature vector that includes both the precomputed and on-demand features that you requested from the API
# MAGIC 
# MAGIC ### At Training Time
# MAGIC At training time, Tecton makes it easy to run the exact same transformation logic against your historical data.  Specifically, Tecton will turn your python transformation into a UDF that can efficiently run your transformation logic against large datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Tutorial: Building an On-Demand Feature
# MAGIC 
# MAGIC In this tutorial, we'll walk through how you can build an use an On-Demand Feature.  
# MAGIC 
# MAGIC **The Feature: ** Is the amount (in dollars) of the transaction a user just made larger than the average transaction they've made in the last 72 hours?

# COMMAND ----------

# Setup
import tecton
import pandas as pd
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### First Draft
# MAGIC Let's start by building the candidate logic in our notebook so we can quickly test it out.  Let's list the key components:
# MAGIC * **Inputs**: On-Demand features can have two types of inputs -- request time data (here this will be the amount of the current transaction), and precomputed features (here we'll use the features computed by [this feature view](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/features/stream_window_aggregate_feature_views/user_transaction_amount_metrics.py)).  The inputs are passed in as pandas DataFrames.
# MAGIC * **Outputs**: On-Demand features are expected to produce a pandas DataFrame as an output.  Here our output will be a flag: **1** if the amount of the transaction is larger than the average, **0** if it is not.
# MAGIC 
# MAGIC **note**: all imports must be done within the transformation logic, which is why we'll import pandas on the first line of the function.

# COMMAND ----------

def transaction_amount_is_higher_than_3_day_average(transaction_request: pandas.DataFrame, user_transaction_amount_metrics: pandas.DataFrame):
    import pandas as pd

    # This column is a feature in the 'user_transaction_amount_metrics' Feature View.
    # The feature values are null if there are no transactions in the 24h window so here we fill the nulls with 0.
    user_transaction_amount_metrics['amount_mean_72h_10m'] = user_transaction_amount_metrics['amount_mean_72h_10m'].fillna(0)

    df = pd.DataFrame()
    df['transaction_amount_is_higher_than_average'] = transaction_request['amount'] > user_transaction_amount_metrics['amount_mean_72h_10m']
    return df

# COMMAND ----------

amount = pd.DataFrame({'amount': [1000]})
historical_metrics = pd.DataFrame({'amount_mean_72h_10m': [750]})
display(transaction_amount_is_higher_than_3_day_average(amount, historical_metrics))

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good!
# MAGIC 
# MAGIC ## Defining an OnDemandFeatureView
# MAGIC Now that we've written our transformation logic, all that's left is defining the feature view to Tecton.  At this point you should already know how most feature views are defined, so lets focus on the new things you'll need to learn
# MAGIC 
# MAGIC ### RequestDataSource
# MAGIC Tecton expects the data passed to an on-demand feature view to have a specified schema.  To describe this schema, you'll use a `RequestDataSource`.  Since our feature expects one field (the amount of the transaction), it will look like this:
# MAGIC 
# MAGIC ```python
# MAGIC # Schema of the input to the OnDemandFeatureView
# MAGIC request_schema = StructType()
# MAGIC request_schema.add(StructField('amount', DoubleType()))
# MAGIC transaction_request = RequestDataSource(request_schema=request_schema)
# MAGIC ```
# MAGIC 
# MAGIC ### Output Schema
# MAGIC Tecton also expects you to specify the schema of the output of your transformation -- here the output is the 1/0 flag we described above, so the schema will look like this:
# MAGIC ```py
# MAGIC # Schema of the output feature value(s)
# MAGIC output_schema = StructType()
# MAGIC output_schema.add(StructField('transaction_amount_is_higher_than_average', BooleanType()))
# MAGIC ```
# MAGIC 
# MAGIC ### On-Demand Feature View Decorator
# MAGIC Finally, just like all other feature views, you'll need to decorate your transformation logic with a decorator that puts everything together. The biggest difference between this decorator and the others you've seen will be that there is no orchestration information in this definition -- since on-demand features aren't precomputed, theres no need to specify things like batch schedules or clusters sizes here.
# MAGIC 
# MAGIC ```py
# MAGIC @on_demand_feature_view(
# MAGIC     inputs={
# MAGIC         'transaction_request': Input(transaction_request),
# MAGIC         'user_transaction_amount_metrics': Input(user_transaction_amount_metrics)
# MAGIC     },
# MAGIC     mode='pandas',
# MAGIC     output_schema=output_schema,
# MAGIC     description='The transaction amount is higher than the 3 day average.'
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ### Putting it all together
# MAGIC 
# MAGIC Here's the full feature definition:
# MAGIC 
# MAGIC ```py
# MAGIC from tecton import RequestDataSource, on_demand_feature_view, Input
# MAGIC from pyspark.sql.types import BooleanType, DoubleType, StructType, StructField
# MAGIC from fraud.features.stream_window_aggregate_feature_views.user_transaction_amount_metrics import user_transaction_amount_metrics
# MAGIC import pandas
# MAGIC 
# MAGIC # Schema of the input to the OnDemandFeatureView
# MAGIC request_schema = StructType()
# MAGIC request_schema.add(StructField('amount', DoubleType()))
# MAGIC transaction_request = RequestDataSource(request_schema=request_schema)
# MAGIC 
# MAGIC # Schema of the output feature value(s)
# MAGIC output_schema = StructType()
# MAGIC output_schema.add(StructField('transaction_amount_is_higher_than_average', BooleanType()))
# MAGIC 
# MAGIC @on_demand_feature_view(
# MAGIC     inputs={
# MAGIC         'transaction_request': Input(transaction_request),
# MAGIC         'user_transaction_amount_metrics': Input(user_transaction_amount_metrics)
# MAGIC     },
# MAGIC     mode='pandas',
# MAGIC     output_schema=output_schema,
# MAGIC     description='The transaction amount is higher than the 3 day average.'
# MAGIC )
# MAGIC def transaction_amount_is_higher_than_3_day_average(transaction_request: pandas.DataFrame, user_transaction_amount_metrics: pandas.DataFrame):
# MAGIC     import pandas as pd
# MAGIC 
# MAGIC     # This column is a feature in the 'user_transaction_amount_metrics' Feature View.
# MAGIC     # The feature values are null if there are no transactions in the 72h window so here we fill the nulls with 0.
# MAGIC     user_transaction_amount_metrics['amount_mean_72h_10m'] = user_transaction_amount_metrics['amount_mean_72h_10m'].fillna(0)
# MAGIC 
# MAGIC     df = pd.DataFrame()
# MAGIC     df['transaction_amount_is_higher_than_average'] = transaction_request['amount'] > user_transaction_amount_metrics['amount_mean_72h_10m']
# MAGIC     return df
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Apply your feature to Tecton
# MAGIC Now that we have a working feature definition, you can add it to a file in your feature repo, and use `tecton apply` to push that feature Tecton. If you don't know how to do this yet, make sure to run through the Tecton Tutorial notebook to learn more.
# MAGIC 
# MAGIC *If you want to skip this step*, you can use [this very similar feature](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/features/on_demand_feature_views/transaction_amount_is_higher_than_average.py) called `transaction_amount_is_higher_than_average` for the rest of the tutorial.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Retrieve Historical Feature Values
# MAGIC To retrieve historical feature values (for example when generating training data), you'll use the same `get_historical_features` API you've used for all other data retrieval. The only new thing you'll need is to include the historical request-time data with your spine, as shown below with the `amount` column:

# COMMAND ----------

fv = tecton.get_feature_view('transaction_amount_is_higher_than_average')

# COMMAND ----------

spine = pd.DataFrame({'timestamp': [datetime(2021, 11, 1), datetime(2021, 11, 2)], 'amount': [5000, 1000], 'user_id': ['C805277159', 'C24965104']})
display(spine)

# COMMAND ----------

display(fv.get_historical_features(spine, from_source=False).to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Fetch Features Online
# MAGIC To retrieve features online, you'll use the same `get_feature_vector` API as before that invokes Tecton's feature serving REST endpoint.  You'll pass in the request-time data via the `request_context_map`, here specifying the `amount` of the current transaction.

# COMMAND ----------

feature_vector = fv.get_online_features({'user_id': 'C805277159'}, request_data={"amount": 5000})
display(feature_vector.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC Or you can directly query the feature serving Rest endpoint:
# MAGIC 
# MAGIC ```
# MAGIC curl -X POST https://[your-cluster].tecton.ai/api/v1/feature-service/get-features\
# MAGIC      -H "Authorization: Tecton-key $TECTON_API_KEY" -d\
# MAGIC '{
# MAGIC   "params": {
# MAGIC     "feature_service_name": "fraud_detection_feature_service",
# MAGIC     "join_key_map": {
# MAGIC       "user_id": "C805277159"
# MAGIC     },
# MAGIC     "request_context_map": {
# MAGIC       "amount": 5000
# MAGIC     },
# MAGIC     "workspace_name": "prod"
# MAGIC   }
# MAGIC }'
# MAGIC ```

# COMMAND ----------


