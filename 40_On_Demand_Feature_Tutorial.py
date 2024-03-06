# Databricks notebook source
# MAGIC %md
# MAGIC #On-Demand Feature Tutorial
# MAGIC
# MAGIC Tecton has 3 basic types of Feature Views:
# MAGIC - [Batch Feature View](https://docs.tecton.ai/docs/defining-features/feature-views/batch-feature-view)
# MAGIC - [Stream Feature View](https://docs.tecton.ai/docs/defining-features/feature-views/stream-feature-view)
# MAGIC - [On-Demand Feature View](https://docs.tecton.ai/docs/defining-features/feature-views/on-demand-feature-view)
# MAGIC
# MAGIC In this tutorial we'll focus on **On-Demand Feature Views**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is an On-Demand Feature?
# MAGIC
# MAGIC Most of the features that you'll build in Tecton are **precomputed** -- this means that Tecton will run the data pipelines needed to compute these features before they are needed, and your ML applications will simply look up precomputed feature values from Tecton.
# MAGIC
# MAGIC In some scenarios, the model of precomputing features doesn't make sense, and instead you'd rather compute the value of a feature **on-demand**.  Some examples:
# MAGIC * You need access to data that is only available just before you need to make a prediction
# MAGIC   * (example) a user is making a transaction, and you want to compute features about the transaction
# MAGIC   * (example) a user just filled out a form in your application, and you want to featurize the data they entered
# MAGIC * Precomputing features is inefficient because most of the features will never be used
# MAGIC   * (example) you want to calculate two users mutual friends, but precomputing mutual friends for every user is infeasible
# MAGIC
# MAGIC For these scenarios, Tecton has support for **On-Demand Features** -- features that are dynamically computed when requesting features for inference.  Also note that inputs for On-Demand Feature Views can be provided on the request to Tecton for feature data, as well as data retrieved from the feature store.
# MAGIC
# MAGIC ## How do they work?
# MAGIC
# MAGIC ### Writing On-Demand Features / Modes Available
# MAGIC On-Demand Features are written in declaritive code just like all other features in Tecton.  They are written in python or pandas code depending on the code specified in the decorator.
# MAGIC
# MAGIC ### At Inference Time
# MAGIC At inference time, the transformation logic for on-demand feature are run directly on the Tecton-managed serving infrastructure. Tecton has developed an efficient method to quickly invoke python functions at serving time without inducing significant overhead. How this works:
# MAGIC
# MAGIC 1. When you invoke [Tecton's Feature Serving API](https://docs.tecton.ai/docs/reading-feature-data/reading-feature-data-for-inference), you'll include any request-time data that needs to be processed in one-or-more on-demand features.
# MAGIC 2. While Tecton is looking up any precomputed features, Tecton will also invoke your on-demand transformation logic to compute the on-demand feature on the fly.
# MAGIC 3. Tecton will return a feature vector that includes both the precomputed and on-demand features that you requested from the API
# MAGIC
# MAGIC ### At Training Time
# MAGIC At training time, Tecton makes it easy to run the exact same transformation logic against your historical data.  Specifically, Tecton will turn your python transformation into a UDF that can efficiently run your transformation logic against large datasets.
# MAGIC
# MAGIC #### Speed
# MAGIC Note that this is your own code and its efficiency can affect serving latency.  Also note there are two supported modes; `python` and `pandas` - the former is quickest for real-time serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tutorial: Building an On-Demand Feature
# MAGIC
# MAGIC In this tutorial, we'll walk through how you can build an use an On-Demand Feature.  
# MAGIC
# MAGIC **The Feature:** Is the amount (in dollars) of the transaction a user just made larger than the average transaction they've made in the last 72 hours?  We'll return a boolean of True/False.

# COMMAND ----------

import tecton
import pandas as pd
from datetime import datetime

# connect to prod workspace or your own workspace
ws = tecton.get_workspace('miket_prod')

# COMMAND ----------

# MAGIC %md
# MAGIC ### First, Test Our Logic
# MAGIC Let's start by testing our logic.  We can do this with a python function and some simulated inputs.  To make porting this to Tecton easier, we can leverage the existing name of values from the feature store as well.  In this case, the 72 hour average is going to be the feature `amt_mean_3d_10m` coming from the Streaming Feature View `user_transaction_amount_metrics` as declared in [this code](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/features/stream_features/user_transaction_amount_metrics.py).  We'll simulate the values of $1000 and $750 below for the transaction being attempted and the 72 hour average retrieved from the feature store respectfully.

# COMMAND ----------

transaction_request = {'amt': 1000} # value provided on the transaction
user_transaction_amount_metrics = {'amt_mean_3d_10m': 750} # value that will be retrieved from the feature store given a user_id

# COMMAND ----------

def transaction_amount_is_higher_than_3d_average(transaction_request, user_transaction_amount_metrics):
    amount_mean = 0 if user_transaction_amount_metrics['amt_mean_3d_10m'] is None else user_transaction_amount_metrics['amt_mean_3d_10m']
    return {'transaction_amount_is_higher_than_3d_average': transaction_request['amt'] > amount_mean}

# COMMAND ----------

transaction_amount_is_higher_than_3d_average(transaction_request, user_transaction_amount_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC Great, the code is looking good.  Let's try it within Tecton's framework.
# MAGIC
# MAGIC ### Declaring Request Input and ODFV Output Schemas
# MAGIC
# MAGIC This feature is going to need two inputs.
# MAGIC 1. The `amt_mean_3d_10m` feature from the `user_transaction_amount_metrics` Streaming Feature View (transaction amount average over the past 72 hour period, made available every 10 minutes)
# MAGIC 2. The amount of the transaction the user is trying to make right now
# MAGIC
# MAGIC The first will come from the feature store, and the latter will be provided on the request for Tecton to enrich the data.  We'll need the `user_id` to look up the first feature, and expect the value for `amt` to be provided on the request.  
# MAGIC
# MAGIC We also need to declare the schema of our output feature.  In this case it is a boolean for whether the current transaction is greater than the average over the past 72 hours.  Note this output includes the feature name as well.
# MAGIC
# MAGIC Below, we'll use Tecton types to declare what the input request schema provides and what the output schema looks like.  We will also declare a RequestSource data source for the input.

# COMMAND ----------

from tecton import RequestSource
from tecton.types import Float64, Field, Bool

request_schema = [Field('amt', Float64)]
transaction_request = RequestSource(schema=request_schema)
output_schema = [Field('transaction_amount_is_higher_than_3d_average', Bool)]

# COMMAND ----------

# MAGIC %md
# MAGIC We can use existing Tecton objects from the server by importing them in for use within the scope of our development notebook.  We'll need to retrieve `user_transaction_amount_metrics` for that.  We can also add the Tecton decorator to the request.  Note the mode is **python**.  In python mode, the input and output schemas will both be dictionaries.

# COMMAND ----------

from tecton import on_demand_feature_view

user_transaction_amount_metrics = ws.get_feature_view('user_transaction_amount_metrics')

@on_demand_feature_view(
    sources=[transaction_request, user_transaction_amount_metrics],
    mode='python',
    schema=output_schema,
)
def transaction_amount_is_higher_than_3d_average(transaction_request, user_transaction_amount_metrics):
    amount_mean = 0 if user_transaction_amount_metrics['amt_mean_3d_10m'] is None else user_transaction_amount_metrics['amt_mean_3d_10m']
    return {'transaction_amount_is_higher_than_3d_average': transaction_request['amt'] > amount_mean}

transaction_amount_is_higher_than_3d_average.validate() # validate the above ODFV declaration looks good to use and register for use within notebook

# COMMAND ----------

# MAGIC %md
# MAGIC There are now several ways we can test this ODFV.  One is by providing it mock inputs and using the `run` function.

# COMMAND ----------

transaction_amount_is_higher_than_3d_average.run(transaction_request={'amt': 500}, user_transaction_amount_metrics={'amt_mean_3d_10m': 800})

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use get_historical_features to pull some values from the online store.  In this case we'll need to provide 3 values - the user_id, amt, and a timestamp to travel to, even if it is right now.  Keep in mind the data is being retrieved from the offline store!  Although this feature view is available as of every wall clock 10 minutes from a streaming source, the offline store itself is loaded on a daily basis.  Keep this in mind when setting a timestamp.  Eg. for this 3 day feature we can use a timestamp of right now and still get values from days ago.  If this was a 30 minute feature instead of 3 days - then the offline store wouldn't have been loaded yet, and we'd have to travel back further.  Let's make a spine and run 'Get Historical Features' (GHF) on the ODFV to see the values.

# COMMAND ----------

spine = pd.DataFrame.from_records([{
    "user_id": "user_934384811883",
    "merchant": "fraud_Rempel Inc",
    "amt": 135.72,
    "timestamp": datetime.utcnow(),
},
{
    "user_id": "user_402539845901",
    "merchant": "fraud_Shanahan-Lehner",
    "amt": 227,
    "timestamp": datetime.utcnow(),
}])

display(spine)

# COMMAND ----------

display(transaction_amount_is_higher_than_3d_average.get_historical_features(spine).to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC While it's great we got the values, it would be nice to see the input from our Streaming Feature View (SFV) as well.  How can we see that?  We can construct a feature service on the fly, and use it to see the input data as well.

# COMMAND ----------

odfv_test_fs = tecton.FeatureService(
  name = 'odfv_test_fs',
  features = [
    user_transaction_amount_metrics[['amt_mean_3d_10m']],
    transaction_amount_is_higher_than_3d_average
  ]
)

odfv_test_fs.validate()

# COMMAND ----------

display(odfv_test_fs.get_historical_features(spine).to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC We can use the `run` function to test it with data from the online store as well, if we pull the input data with `get_online_features` first.  The latter method will not be supported within the scope of a notebook declared object; the ODFV must be registered within the Tecton workspace to allow it to read from the online store itself.  So we can first look up our input data `amt_mean_3d_10m` for a given user from the online store.

# COMMAND ----------

online_amt_mean_3d_10m = ws.get_feature_view('user_transaction_amount_metrics').get_online_features(join_keys={'user_id': 'user_402539845901'}).to_dict()['amt_mean_3d_10m']
online_amt_mean_3d_10m

# COMMAND ----------

# MAGIC %md
# MAGIC It can then be passed to `run` as an input similarly as before, but with the value passed in.  In this case we've retrieved some data from the online store and are using it in our mock input for the run below which happens in our local cluster.

# COMMAND ----------

transaction_amount_is_higher_than_3d_average.run(transaction_request={'amt': 125}, user_transaction_amount_metrics={'amt_mean_3d_10m': online_amt_mean_3d_10m})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registering the Feature to Tecton
# MAGIC
# MAGIC We can proceed to register the feature in Tecton.  Note for it to be callabe online there, it will need to be registered to a live workspace with materialization to the online store, like the `prod` workspace.  This can be done in the trial environment, and note the feature itself does not create any new materialization jobs.  Also note however when using the live prod workspace rather than your own, feature churn could be occurring if multiple evaluating colleagues are applying different versions of the repo, as no code review or CI/CD process with collaboratively managed code is required in the context of the tutorials or evaluation environment.  Keep this in mind if the are other materialized features in the prod workspace, so that you are not deleting features colleagues have worked on.  This can be done by coping those same features into your copy of the repo.
# MAGIC
# MAGIC To add this feature to the repo, create the following file: `fraud/features/on_demand_feature_views/3d_odfv.py` and populate it with the below code.
# MAGIC
# MAGIC ```python
# MAGIC from tecton import RequestSource, on_demand_feature_view
# MAGIC from tecton.types import Float64, Field, Bool
# MAGIC from fraud.features.stream_features.user_transaction_amount_metrics import user_transaction_amount_metrics
# MAGIC
# MAGIC request_schema = [Field('amt', Float64)]
# MAGIC transaction_request = RequestSource(schema=request_schema)
# MAGIC output_schema = [Field('transaction_amount_is_higher_than_3d_average', Bool)]
# MAGIC
# MAGIC @on_demand_feature_view(
# MAGIC     sources=[transaction_request, user_transaction_amount_metrics],
# MAGIC     mode='python',
# MAGIC     schema=output_schema,
# MAGIC )
# MAGIC def transaction_amount_is_higher_than_3d_average(transaction_request, user_transaction_amount_metrics):
# MAGIC     amount_mean = 0 if user_transaction_amount_metrics['amt_mean_3d_10m'] is None else user_transaction_amount_metrics['amt_mean_3d_10m']
# MAGIC     return {'transaction_amount_is_higher_than_3d_average': transaction_request['amt'] > amount_mean}
# MAGIC ```
# MAGIC
# MAGIC and run `tecton apply`.

# COMMAND ----------

ws.get_feature_view('transaction_amount_is_higher_than_3d_average').get_online_features(join_keys={'user_id': 'user_402539845901'}, request_data={'amt': 125}).to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ### This can be created in pandas too!
# MAGIC
# MAGIC Note this can also be constructed in **pandas** mode.  While the schema is defined similarly, now the input and output objects for the python function are _dataframes_.  The code looks as follows:
# MAGIC
# MAGIC ```python
# MAGIC from tecton import RequestSource, on_demand_feature_view
# MAGIC from tecton.types import Float64, Field, Bool
# MAGIC from fraud.features.stream_features.user_transaction_amount_metrics import user_transaction_amount_metrics
# MAGIC
# MAGIC request_schema = [Field('amt', Float64)]
# MAGIC transaction_request = RequestSource(schema=request_schema)
# MAGIC output_schema = [Field('transaction_amount_is_higher_than_3d_average', Bool)]
# MAGIC
# MAGIC @on_demand_feature_view(
# MAGIC     sources=[transaction_request, user_transaction_amount_metrics],
# MAGIC     mode='pandas',
# MAGIC     schema=output_schema,
# MAGIC )
# MAGIC def transaction_amount_is_higher_than_3d_average(transaction_request: pandas.DataFrame, user_transaction_amount_metrics: pandas.DataFrame):
# MAGIC     import pandas as pd
# MAGIC
# MAGIC     user_transaction_amount_metrics['amt_mean_3d_10m'] = user_transaction_amount_metrics['amt_mean_3d_10m'].fillna(0)
# MAGIC
# MAGIC     df = pd.DataFrame()
# MAGIC     df['transaction_amount_is_higher_than_3d_average'] = transaction_request['amt'] > user_transaction_amount_metrics['amt_mean_3d_10m']
# MAGIC     return df
# MAGIC ```
# MAGIC
# MAGIC **This does come with the cost of additional time to instantiate pandas and work with dataframes.**  Arguably it should only be used for offline / batch inference use cases.

# COMMAND ----------



# COMMAND ----------

