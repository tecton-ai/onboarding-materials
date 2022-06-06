# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #Stream Window Aggregate Feature View (SWAF) Tutorial
# MAGIC 
# MAGIC Tecton has 5 basic types of Feature Views:
# MAGIC - [Batch Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_feature_view.html)
# MAGIC - [Batch Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_window_aggregate_feature_view.html)
# MAGIC - [Stream Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/stream_feature_view.html)
# MAGIC - [Stream Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/stream_window_aggregate_feature_view.html)
# MAGIC - [On-Demand Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/on_demand_feature_view.html)
# MAGIC 
# MAGIC In this tutorial we'll focus on **Stream Window Aggregate Feature View**
# MAGIC 
# MAGIC 
# MAGIC **This tutorial should take about 20 minutes to complete**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## What is a Stream Window Aggregate Feature View?
# MAGIC In our experience, a huge portion of the features used for production ML models are simple time-window aggregations like:
# MAGIC 
# MAGIC * How many transactions has a user made in the last 30 minutes
# MAGIC * How many times has a product been purchased in the last day
# MAGIC * How many orders have been placed at a restaurant in the last hour
# MAGIC 
# MAGIC In many cases, these aggregations need to be very __fresh__ -- its critical to know very up-to-date values of these features.
# MAGIC * If you're doing transaction fraud, you need to recognize new patterns within seconds
# MAGIC * If you're predicting ETAs for food delivery, you need to know how many pending orders there are from a restaurant
# MAGIC 
# MAGIC In order to have low-latency, very fresh values for these features, most people turn to streaming technologies like Apache Kafka.  Unfortunately, writing these aggregations the most intuitive way (using a `GROUP BY` for example), ends up being very computationally expensive -- many organization struggle to write efficient implementations of those features.  Further, it can be incredibly difficult to have online and offline parity of these models, since typically models are trained from historical data, but inference is done from data on a stream.
# MAGIC 
# MAGIC Tecton solves these issues by providing a special type of feature that can efficiently calculate time-window aggregates on a stream, called a [Streaming Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/stream_window_aggregate_feature_view.html).  Further, Tecton can apply these transformations automatically against historical data so that you can train models without worrying about online / offline parity.
# MAGIC 
# MAGIC For more info, you can also [check out our techincal blog about low-latency aggregations](https://www.tecton.ai/blog/real-time-aggregation-features-for-machine-learning-part-1/).
# MAGIC 
# MAGIC **Let's check out how to build one of these features, check it out below:**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Setting up the data source
# MAGIC 
# MAGIC Streaming features depend on streaming data sources
# MAGIC 
# MAGIC At the moment we support two options:
# MAGIC - Kinesis Streams
# MAGIC - Kafka
# MAGIC 
# MAGIC For this tutorial we'll leverage a pre-configured Kinesis data stream
# MAGIC 
# MAGIC ###Let's exlore that stream...

# COMMAND ----------

import tecton # this is how we pull in the Tecton SDK

tecton.list_data_sources() #lets list out the datasources available to us



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We'll be working with the **transactions_stream**
# MAGIC 
# MAGIC Lets query it to see what data we have access to...

# COMMAND ----------

transactions_stream = tecton.get_data_source('transactions_stream')
transactions_stream.start_stream_preview("temp_table")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Please note that the **start_stream_preview** command should only be used in a testing environment for short periods of time. The temp_table will grow inifinitely large (memory and time permitting).
# MAGIC 
# MAGIC For additional methods that can be run on stream data source please visit our docs: https://docs.tecton.ai/api-reference/stubs/tecton.interactive.StreamDataSource.html#
# MAGIC 
# MAGIC Lets query the newly created **temp_table**

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from temp_table limit (10)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Once queried please go back and cancel the spark job from cell 5.
# MAGIC 
# MAGIC From the query above we can gather that we have a constant stream of customer transactions. The customer id can be found in the **nameOrig** column. When we define the feature below we'll use this column as the primary key. Addtionally we have a **timestamp** column that will also be useful to us when we define the Feature View.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Definining a Feature View
# MAGIC 
# MAGIC Lets take a look at the underlying code to get a sense for how to define a Stream Window Aggregate Feature View. Please review the inline comments.
# MAGIC 
# MAGIC ```
# MAGIC @stream_window_aggregate_feature_view(
# MAGIC     inputs={'transactions': Input(transactions_stream)},     # Input here is the kinesis stream datasource that we've configured in Tecton
# MAGIC     entities=[user], 
# MAGIC     mode='spark_sql',                                        # Can be spark_sql or pyspark
# MAGIC     aggregation_slide_period='10m',                   # Slide period continous indicates we went up to the second aggregates
# MAGIC     aggregations=[
# MAGIC         FeatureAggregation(column='counter',
# MAGIC                            function='count',
# MAGIC                            time_windows=['1h', '12h', '24h'] # This is where we define the aggregation periods in this case 1 hour, 12 hour and 24 hours
# MAGIC                            )                                 # Any other aggregations we'd like e.g. sum, mean can be added as additional FeatureAggregation configs
# MAGIC     ],                                                       # Docs: https://docs.tecton.ai/api-reference/stubs/tecton.FeatureAggregation.html#tecton.FeatureAggregation                                     
# MAGIC                                                                                                   
# MAGIC     online=True,                                             # Yes to materializing the feature into a low latency data store
# MAGIC     offline=True,                                            # Yes to materializing historical data (backfilling) 
# MAGIC     feature_start_time=datetime(2021, 6, 1),                 # Start date of the feature for aggregation purposes
# MAGIC     family='fraud',
# MAGIC     tags={'release': 'production'},
# MAGIC     owner='eddie@tecton.ai',
# MAGIC     description='Number of transactions'
# MAGIC )
# MAGIC 
# MAGIC # this is where we define the transformation, as noted above it is spark_sql
# MAGIC def continuous_customer_transaction_count(transactions):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             nameorig as user_id,                             # this is the primary key we noted earlier when querying the data source
# MAGIC             1 as counter,                                    # counter will be the name of column we supply our model
# MAGIC             timestamp 
# MAGIC         FROM
# MAGIC             {transactions}
# MAGIC         '''
# MAGIC 
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Creating a Feature in Tecton
# MAGIC 
# MAGIC We'll have to go back to a terminal with the Tecton CLI configured in order to do this. Please refer to these docs for installing and configuring the CLI: https://docs.tecton.ai/v2/setting-up-tecton/02-tecton-cli-setup.html
# MAGIC 
# MAGIC Once configured you can clone the following github repo to bring in the Feature View's we'll create: https://github.com/tecton-ai-ext/tecton-sample-repo
# MAGIC 
# MAGIC **Important to note that github is not a requirement for creating Feature Views (or any other primitive) in Tecton.**
# MAGIC 
# MAGIC The file structure of the repo is as follows:
# MAGIC ```
# MAGIC /
# MAGIC |- ads
# MAGIC |- fraud
# MAGIC |  |- data_sources
# MAGIC |  |- feature_services
# MAGIC |  |- features
# MAGIC |   |- tests
# MAGIC |   |- batch_feature_views
# MAGIC |   |- batch_window_aggregate_feature_views
# MAGIC |   |- stream_feature_views
# MAGIC |   |- stream_window_aggregate_feature_views
# MAGIC      - continuous_customer_transaction_count.py
# MAGIC 
# MAGIC |   |- on_demand_feature_views
# MAGIC |   |- feature_tables
# MAGIC    - entities.py
# MAGIC ```
# MAGIC 
# MAGIC The Feature View we described above is under **stream_window_aggregate_feature_views** specifically the ***continuous_customer_transaction_count.py*** file.
# MAGIC 
# MAGIC **One more note: This structure shows an example of how a Feature Repository can be organized. It includes different folders for different use cases (Ad Prediction and Fraud Detection). We have organized the repo according to our best practices, but you can use whatever directory structure you want. There are no restrictions on how files or directories are organized within a Feature Repository.**
# MAGIC 
# MAGIC Now that we've built our new feature within this file, all we need to do is run `$ tecton apply` to submit all of the manifests to the newly created workspace. We should see output similar to this:
# MAGIC 
# MAGIC ```
# MAGIC $ tecton apply
# MAGIC Using workspace "<your new workspace>" on cluster https://staging.tecton.ai
# MAGIC ✅ Imported 29 Python modules from the feature repository
# MAGIC ✅ [Plan Hooks] Tests passed!
# MAGIC ✅ Collecting local feature declarations
# MAGIC ✅ Performing server-side validation of feature declarations
# MAGIC  ↓↓↓↓↓↓↓↓↓↓↓↓ Plan Start ↓↓↓↓↓↓↓↓↓↓
# MAGIC 
# MAGIC   + Create StreamDataSource
# MAGIC     name:            ad_impressions_stream
# MAGIC 
# MAGIC   + Create BatchDataSource
# MAGIC     name:            ad_impressions_batch
# MAGIC 
# MAGIC   .
# MAGIC   .
# MAGIC   .
# MAGIC 
# MAGIC   + Create FeatureService
# MAGIC     name:            fraud_detection_feature_service
# MAGIC     owner:           matt@tecton.ai
# MAGIC     description:     A FeatureService providing features for a model that predicts if a transaction is fraudulent.
# MAGIC 
# MAGIC  ↑↑↑↑↑↑↑↑↑↑↑↑ Plan End ↑↑↑↑↑↑↑↑↑↑↑↑
# MAGIC Are you sure you want to apply this plan? [y/N]>
# MAGIC ```
# MAGIC 
# MAGIC Once this is done we will have the Feature View we reviewed earlier in Tecton: **continuous_customer_transaction_count**
# MAGIC 
# MAGIC Back in this notebook we can run the cell below to get a specific vector from Tecton for this Feature View.

# COMMAND ----------

fv = tecton.get_feature_view('continuous_customer_transaction_count')

#we can grab a value from the nameOrig column above and bring up a specific vector
print(fv.get_feature_vector({"user_id": "C1834792885"}).to_dict())
