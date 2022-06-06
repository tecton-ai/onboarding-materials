# Databricks notebook source
# MAGIC %md
# MAGIC # 📚 Batch Feature Views
# MAGIC 
# MAGIC In this tutorial notebook, you will:
# MAGIC 
# MAGIC 1. Create a Batch Feature View with multiple features
# MAGIC 2. Pull results from the feature store for a specified set of entities
# MAGIC 3. Create a Batch Window Aggregate Feature View with multiple aggregate features
# MAGIC 4. Construct a set of independent Transformations to pipeline together into features
# MAGIC 5. Construct the same feature all in inline code
# MAGIC 
# MAGIC At the end of this tutorial, you should have a more full understanding of Batch Feature Views in Tecton.
# MAGIC 
# MAGIC Helpful links (Click the blue Sign In button):
# MAGIC - Tecton Web UI: [_your-cluster-name_].tecton.ai
# MAGIC - <a href="https://docs.tecton.ai/v2/" target="_blank">Tecton Docs</a>
# MAGIC - <a href="https://s3-us-west-2.amazonaws.com/tecton.ai.public/documentation/tecton-py/index.html" target="_blank">Tecton Python SDK Docs</a>

# COMMAND ----------

import tecton

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
# MAGIC #### What is a Batch Feature View (BFV)?
# MAGIC [Batch Feature Views](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_feature_view.html) are used for single record transforms or for more custom defined aggregate logic not available out of the box with Batch Window Aggregate Feature Views.
# MAGIC 
# MAGIC #### When should I use a Batch Feature View?
# MAGIC Batch Feature Views are built based on historical data or data coming from at rest sources.  Typically these are databases or data warehouses, and data lakes or object storage.  The smallest supported scheduling window is currently hourly, for intra-day batching.
# MAGIC 
# MAGIC #### Best Practice: Feature View Consolidation
# MAGIC Consolidating features into a single Batch Feature View wherever appropriate rather than defining separate views is a good best practice to follow.  At the beginning of the intro tutorial, a user_has_good_credit feature exists, and a user_has_great_credit feature is defined in a separate Feature View.  These work on the same data and entity id, and could both be defined in a single view, creating many benefits.
# MAGIC - A single Feature View to define and maintain
# MAGIC - Lower costs from running a single cluster each materialization job
# MAGIC - A single extract/pass from the source data
# MAGIC 
# MAGIC _Exception_: It may be benefecial to separate out slowly changing from more quickly changing data.  Assume 200 customer demographic fields where 180 are fairly static, and 20 that may change monthly or weekly.  Defining two Feature Views in this case may provide benefit.  The static job would be expected to write few records, and the job that handles the more likely to change 20 features would improve by not moving and repeatedly writing 180 unchanged features.  Note: to accomplish this specific scenario, an updated timestamp outside of Tecton _respective to each set being updated_ would need to be available.  If only one updated timestamp were available, Tecton would not itself be able to distinguish between either set, and all new records would always be moved and written.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Consolidated Batch Feature View: user_credit_categories
# MAGIC 
# MAGIC Similar to the tutorial, we could create a `fraud/features/batch_feature_views/user_has_bad_credit.py`, in parallel in the repo with `fraud/features/batch_feature_views/user_has_good_credit_sql.py` and `fraud/features/batch_feature_views/user_has_great_credit.py`.  However, we would be far better off removing both of those Feature Views and consolidating all 3 features (bad, good, great) into a new one.  Create a `fraud/features/batch_feature_views/user_credit_categories.py` to provide all metrics.  Populate it with the following:
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton import batch_feature_view, Input, DatabricksClusterConfig, BackfillConfig
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.credit_scores_batch import credit_scores_batch
# MAGIC from datetime import datetime
# MAGIC 
# MAGIC @batch_feature_view(
# MAGIC     # explicit name of the feature view, which will override the default of using the function name below
# MAGIC     name_override='user_credit_categories',
# MAGIC     inputs={'credit_scores': Input(credit_scores_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2021, 7, 1),
# MAGIC     batch_schedule='1d',
# MAGIC     ttl='30days',
# MAGIC     backfill_config=BackfillConfig("multiple_batch_schedule_intervals_per_job")
# MAGIC )
# MAGIC # function name is implicitly going to be used as feature view
# MAGIC def user_credit_categories(credit_scores):
# MAGIC     # define our 3 feature names based on different credit score tiering logic
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             IF (credit_score < 500, 1, 0) as cat_user_has_bad_credit,
# MAGIC             IF (credit_score > 670, 1, 0) as cat_user_has_good_credit,
# MAGIC             IF (credit_score > 740, 1, 0) as cat_user_has_great_credit,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {credit_scores}
# MAGIC         '''
# MAGIC </pre>
# MAGIC 
# MAGIC Run `tecton apply` - we can also take a quick look at this new Feature View and the feature values it contains.

# COMMAND ----------

fv = ws.get_feature_view('user_credit_categories')

# Get a range of feature data - from_source=True required if data not materialized
feature_df = fv.get_historical_features(start_time=datetime(2021, 7, 1), from_source=True)
display(feature_df.to_spark().limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note we can also look up features by entity, like a couple user_id values for example.

# COMMAND ----------

df_filter = spark.createDataFrame(
    [
        ('C1305486145',),  
        ('C1912850431',),
    ],
    ["user_id"]
)

# Get a range of feature data - from_source=True required if data not materialized
feature_df = fv.get_historical_features(start_time=datetime(2021, 7, 1), from_source=True, entities=df_filter)
display(feature_df.to_spark().limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### What is a Batch Window Aggregate Feature View (BWAFV)?
# MAGIC [Batch Window Aggregate Feature Views](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_window_aggregate_feature_view.html) are used for creating aggregates from the same data sources as Batch Feature Views.  Rather than requiring the developer specify code and logic to define the views, the aggregates over the time periods of interest can simply be asked.  This views also take advantage of the intelligent tiling storage and retrieval approached mentioned in our 2-part blog series starting here: [Real-Time Aggregation Features for Machine Learning (Part 1)](https://www.tecton.ai/blog/real-time-aggregation-features-for-machine-learning-part-1/).
# MAGIC 
# MAGIC #### When should I use a Batch Window Aggregate Feature View?
# MAGIC Use these to construct aggregates of data, such as the minimum, maximum, average, etc. of a value over various time windows, like an hour, a day, a week, and so on.  Simply specify the metrics of interest, and Tecton will construct them going forward, as well as backfill their history to any point in time you specify and have data available for.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's make an aggregate view that looks at the past 30 day and 180 day min/max/average for credit scores.  Create a `fraud/features/batch_window_aggregate_feature_views/user_credit_history.py` file.  Paste in the contents below.
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton.feature_views import batch_window_aggregate_feature_view
# MAGIC from tecton.feature_views.feature_view import Input
# MAGIC from tecton import FeatureAggregation
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.credit_scores_batch import credit_scores_batch
# MAGIC from datetime import datetime
# MAGIC 
# MAGIC @batch_window_aggregate_feature_view(
# MAGIC     inputs={'credit_scores': Input(credit_scores_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     aggregation_slide_period='1d',
# MAGIC     aggregations=[FeatureAggregation(column='credit_score', function='min', time_windows=['30d','180d']),
# MAGIC                   FeatureAggregation(column='credit_score', function='max', time_windows=['30d','180d']),
# MAGIC                   FeatureAggregation(column='credit_score', function='mean', time_windows=['30d','180d'])],
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2020, 6, 1)
# MAGIC )
# MAGIC def user_credit_history(credit_scores):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             credit_score,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {credit_scores}
# MAGIC         '''
# MAGIC </pre>
# MAGIC 
# MAGIC Note the aggregates in the `@batch_window_aggregate_feature_view` decorator operate on the data defined in the `user_credit_history` function.  What would typically take a fair amount of developer effort in either python with dataframes or SQL in a data warehouse is made easy for us in Tecton; we are creating multiple aggregates, across multiple windows and types, by simply asking for them.  The `feature_start_time` will also signal Tecton to backfill this for us, so a traditionally challenging training data set to construct will easily now be retrievable from the feature store.  Going forward, both offline and online stores will be filled with data as it comes in, meaning we already have production inference ready pipelines in place with just this code.  This effort takes literally minutes, rather than weeks or months!
# MAGIC 
# MAGIC After a `tecton apply` the following feature pipeline should be available, defining all 6 feature values requested.
# MAGIC 
# MAGIC <img src=https://www.doyouevendata.com/wp-content/uploads/2021/12/Screen-Shot-2021-12-20-at-9.17.16-AM.png width=80%>

# COMMAND ----------

fv = ws.get_feature_view('user_credit_history')

# Get a range of feature data - from_source=True required if data not materialized
feature_df = fv.get_historical_features(start_time=datetime(2021, 7, 1), from_source=True)
display(feature_df.to_spark().limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Pipeline (Spark engine)
# MAGIC 
# MAGIC Features can be created via a _pipeline of transformations_ strung together, applying both transformations and aggregations to data.  Transformations can be defined and used individually and among many features.  An example is as follows below.
# MAGIC 
# MAGIC #### What is a Transformation?
# MAGIC 
# MAGIC A Transformation is a Tecton object that describes a set of operations on data. The operations are expressed through standard frameworks such as Spark SQL, PySpark, and Pandas.  A Transformation is required to create a feature within a Feature view. Once defined, a Transformation can be reused within multiple Feature Views, or multiple Transformations can be composed within a single Feature View.
# MAGIC 
# MAGIC <img src ='https://www.doyouevendata.com/wp-content/uploads/2021/10/Screen-Shot-2021-10-30-at-9.41.36-PM.png' width=80%>
# MAGIC 
# MAGIC In the above image, we're looking at the data pipeline Tecton established for the user_has_great_credit feature.  To understand exactly what these items are: the first demo_fraud.credit_scores is a table available in our hive catalog as batch source data.  The credit_scores_batch is a Tecton Data Source object.  The purple user_has_great_credit contains the transformation logic.  The cyan user_has_great_credit is the feature view which uses it, and creates the single green user_has_great_credit feature, persisted to both offline and online stores.
# MAGIC 
# MAGIC As noted in the intro tutorial, this feature is defined with:
# MAGIC 
# MAGIC <pre>
# MAGIC @batch_feature_view(
# MAGIC     inputs={'credit_scores': Input(credit_scores_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2020, 10, 10),
# MAGIC     batch_schedule='1d',
# MAGIC     ttl='30days',
# MAGIC     backfill_config=BackfillConfig('multiple_batch_schedule_intervals_per_job')
# MAGIC )
# MAGIC def user_has_great_credit(credit_scores):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             IF (credit_score > 740, 1, 0) as user_has_great_credit,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {credit_scores}
# MAGIC         '''
# MAGIC </pre>
# MAGIC 
# MAGIC which is shorthand for:
# MAGIC 
# MAGIC <pre>
# MAGIC @transformation(mode="spark_sql")
# MAGIC def user_has_great_credit_transformation(input_df):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             IF (credit_score > 740, 1, 0) as user_has_great_credit,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {input_df}
# MAGIC         '''
# MAGIC 
# MAGIC @batch_feature_view(
# MAGIC     inputs={'credit_scores': Input(credit_scores_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2020, 10, 10),
# MAGIC     batch_schedule='1d',
# MAGIC     ttl='30days',
# MAGIC     backfill_config=BackfillConfig('multiple_batch_schedule_intervals_per_job')
# MAGIC )
# MAGIC def user_has_great_credit(credit_scores):
# MAGIC     return user_has_great_credit_transformation(credit_scores)
# MAGIC </pre>
# MAGIC 
# MAGIC which makes it more clear where this transformation is coming from.
# MAGIC 
# MAGIC We will construct a series of transformations to ultimately create a 60 day count of weekend evening transactions, inspired from user_weekend_transaction_count_30d.  This will be done with transaction data based on the timestamp.  Let's take a look at this data.

# COMMAND ----------

# get data source
transactions_batch = ws.get_data_source('transactions_batch')
# display records from data source in spark dataframe
display(transactions_batch.get_dataframe().to_spark().limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC We can develop against this data in a couple of different ways.  We can create a dataframe from the data source, as well as register it as a temporary view.

# COMMAND ----------

transactions_batch = ws.get_data_source('transactions_batch')
df_transactions = transactions_batch.get_dataframe().to_spark().limit(100).cache()
transactions_batch.get_dataframe().to_spark().createOrReplaceTempView("transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC #### is_weekend_transform
# MAGIC First, we'll develop an `is_weekend_transform` transformation.  The logic can be developed in pyspark against the dataframe.  A 1 will be returned for a Saturday or Sunday, else a 0.

# COMMAND ----------

# pyspark logic against dataframe
from pyspark.sql.functions import dayofweek, col, to_timestamp
display(
  df_transactions.withColumn("is_weekend", dayofweek(to_timestamp(col('timestamp'))).isin([1,7]).cast("int"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, it can be developed in spark_sql against the temporary view made.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *,
# MAGIC case when dayofweek('timestamp') in (1, 7) then 1 else 0 end as is_weekend
# MAGIC FROM transactions
# MAGIC limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC With the language of your choice, create a `fraud/transformations/is_weekend_transform.py` and populate it.  Using pyspark, the code would be as follows:
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton import transformation
# MAGIC 
# MAGIC @transformation(mode='pyspark')
# MAGIC def is_weekend_transform(input_df, timestamp_column):
# MAGIC     from pyspark.sql.functions import dayofweek, col, to_timestamp
# MAGIC     return input_df.withColumn("is_weekend", dayofweek(to_timestamp(col(timestamp_column))).isin([1,7]).cast("int"))
# MAGIC </pre>
# MAGIC 
# MAGIC Using the Tecton CLI, run `tecton apply` and create the transform in the workspace.  It can now be tested.

# COMMAND ----------

display(ws.get_transformation('is_weekend_transform').run(df_transactions, 'timestamp').to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC #### is_evening_transform
# MAGIC Next, we'll develop an `is_evening_transform` transformation.  This will be a 1 for hours in which a person is typically sleeping, 11pm to 7am, else a 0.  This will be done in Spark SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *,
# MAGIC case when hour('timestamp') < 7 or hour('timestamp') >= 23 then 1 else 0 end as is_evening
# MAGIC FROM transactions
# MAGIC limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC Create a `fraud/transformations/is_evening_transform.py` and populate it, using the following Spark SQL:
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton import transformation
# MAGIC 
# MAGIC # mark a transaction as occurring during the evening if it occurs >= 11pm and before 7am
# MAGIC @transformation(mode='spark_sql')
# MAGIC def is_evening_transform(input_df, timestamp_column):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             *,
# MAGIC             case when hour({timestamp_column}) < 7 or hour({timestamp_column}) >= 23 then 1 else 0 end as is_evening
# MAGIC         FROM
# MAGIC             {input_df}
# MAGIC         '''
# MAGIC </pre>
# MAGIC 
# MAGIC Using the Tecton CLI, run `tecton apply` and create the transform in the workspace.  It can now be tested as well.

# COMMAND ----------

display(ws.get_transformation('is_evening_transform').run(df_transactions, 'timestamp').to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Testing both transforms together
# MAGIC These transforms can both be applied in a chain to add both fields to a final result dataframe.  Both values can be seen added below.

# COMMAND ----------

display(ws.get_transformation('is_evening_transform').run(ws.get_transformation('is_weekend_transform').run(df_transactions, 'timestamp').to_spark(), 'timestamp').to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC #### weekend_evening_transaction_transform transform and user_weekend_evening_transaction_counts 30 and 60 day aggregates batch feature
# MAGIC Next, we'll develop an `weekend_evening_transaction_count_n_days` transformation, and use it in a `user_weekend_evening_transactions_30d` batch feature view which will be defined as a **pipeline of transformations**.  This logic will create a finaly binary 1 for those transactions that are both on the weekend and occurred during the evening, to then sum up for our window period.  We can register the above dataframe as a temporary view to run Spark SQL against it and use what the above transforms created.

# COMMAND ----------

ws.get_transformation('is_evening_transform').run(ws.get_transformation('is_weekend_transform').run(df_transactions, 'timestamp').to_spark(), 'timestamp').to_spark().createOrReplaceTempView("input_df")

# COMMAND ----------

# MAGIC %sql
# MAGIC select amount, nameorig, namedest, timestamp, is_evening, is_weekend,
# MAGIC case when is_weekend = 1 and is_evening = 1 then 1 else 0 end as weekend_evening_transform
# MAGIC from input_df limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC The above logic will be used in a final transform `weekend_evening_transaction_transform` that combines the logic of both.  Of course, all of this could have been defined in a single transformation, however the intent of this example is to show a pipelined chain of multiple transformations.
# MAGIC 
# MAGIC <pre>
# MAGIC @transformation(mode='spark_sql')
# MAGIC def weekend_evening_transaction_transform(input_df):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             nameorig as user_id,
# MAGIC             case when is_weekend = 1 and is_evening = 1 then 1 else 0 end as weekend_evening_transform,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {input_df}
# MAGIC         '''
# MAGIC </pre>
# MAGIC 
# MAGIC Once all of these transformations are complete, the resulting dataframe can simply have counts summed up.  A [Batch Window Aggregate Feature View](https://docs.tecton.ai/v2/overviews/framework/feature_views/batch_window_aggregate_feature_view.html) taking advantage of Tecton's ability to provide native aggregates easily will be leveraged.  We can specify any time windows we would like here; the example below uses 30 and 60 day counts.  This feature view is `user_weekend_evening_transaction_counts` - which you can see uses a mode of _pipeline_ and chains together all of the transforms created above.  Note that further manipulation of the data cannot be defined in the pipeline defined feature; all data changes must take place within the pipeline.
# MAGIC 
# MAGIC Create a `fraud/features/batch_window_aggregate_feature_views/user_weekend_evening_transaction_counts.py` and populate it, using the following Spark SQL:
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton.feature_views import batch_window_aggregate_feature_view
# MAGIC from tecton import transformation, const, FeatureAggregation
# MAGIC from tecton.feature_views.feature_view import Input
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.transactions_batch import transactions_batch
# MAGIC from fraud.transformations.is_evening_transform import is_evening_transform
# MAGIC from fraud.transformations.is_weekend_transform import is_weekend_transform
# MAGIC from datetime import datetime
# MAGIC 
# MAGIC @transformation(mode='spark_sql')
# MAGIC def weekend_evening_transaction_transform(input_df):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             nameorig as user_id,
# MAGIC             case when is_weekend = 1 and is_evening = 1 then 1 else 0 end as weekend_evening_transform,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {input_df}
# MAGIC         '''
# MAGIC 
# MAGIC @batch_window_aggregate_feature_view(
# MAGIC     inputs={'transactions_batch': Input(transactions_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='pipeline',
# MAGIC     aggregation_slide_period='1d',
# MAGIC     aggregations=[FeatureAggregation(column='weekend_evening_transform', function='sum', time_windows=['30d','60d'])],
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2021, 6, 1),
# MAGIC     family='fraud',
# MAGIC     tags={'release': 'production'},
# MAGIC     owner='matt@tecton.ai',
# MAGIC     description='User transaction totals over a series of time windows, updated daily.'
# MAGIC )
# MAGIC def user_weekend_evening_transaction_counts(transactions_batch):
# MAGIC     timestamp_key = const("timestamp")
# MAGIC     return weekend_evening_transaction_transform(
# MAGIC         is_weekend_transform(
# MAGIC             is_evening_transform(transactions_batch, timestamp_key)
# MAGIC             , timestamp_key
# MAGIC         ),
# MAGIC     )
# MAGIC </pre>
# MAGIC 
# MAGIC Using the Tecton CLI, run `tecton apply` and create the transform in the workspace and test.
# MAGIC 
# MAGIC This will create the feature view, data pipelines, and materialization jobs to backfill the feature values.
# MAGIC 
# MAGIC <img src='https://www.doyouevendata.com/wp-content/uploads/2021/11/Screen-Shot-2021-11-22-at-10.46.12-AM.png' width=80%>
# MAGIC 
# MAGIC Here we can see each transform defined, their order of execution, and their ultimate aggregation into 30 and 60 day values.

# COMMAND ----------

fv = ws.get_feature_view('user_weekend_evening_transaction_counts')

# Get a range of feature data - from_source=True required if data not materialized
feature_df = fv.get_historical_features(start_time=datetime(2021, 7, 1), from_source=True)
display(feature_df.to_spark().limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Alternative Logic with Inline Code
# MAGIC 
# MAGIC Note the feature above was created as an example of showing separately defined transformations that could be imported and used across multiple Feature Views.  It is also possible to simply declare all of this code in a series of inline functions and not as Tecton transformation primitives, if all done in psypark as in the example below.
# MAGIC 
# MAGIC Create a `fraud/features/batch_window_aggregate_feature_views/user_weekend_evening_txn_counts_inline.py` and populate it, using the following Spark SQL:
# MAGIC 
# MAGIC <pre>
# MAGIC from tecton.feature_views import batch_window_aggregate_feature_view
# MAGIC from tecton import transformation, const, FeatureAggregation
# MAGIC from tecton.feature_views.feature_view import Input
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.transactions_batch import transactions_batch
# MAGIC from datetime import datetime
# MAGIC 
# MAGIC def is_evening_inline(input_df, timestamp_column):
# MAGIC     from pyspark.sql.functions import hour, col, to_timestamp
# MAGIC     return input_df.withColumn("is_evening", hour(to_timestamp(col(timestamp_column))).isin([0, 1, 2, 3, 4, 5, 6, 23]).cast("int"))
# MAGIC 
# MAGIC def is_weekend_inline(input_df, timestamp_column):
# MAGIC     from pyspark.sql.functions import dayofweek, col, to_timestamp
# MAGIC     return input_df.withColumn("is_weekend", dayofweek(to_timestamp(col(timestamp_column))).isin([1,7]).cast("int"))
# MAGIC 
# MAGIC def is_evening_weekend_inline(input_df):
# MAGIC     from pyspark.sql.functions import when, col
# MAGIC     return input_df.withColumn("is_evening_weekend", when((col('is_evening') == 1) & (col('is_weekend') == 1), 1).otherwise(0))
# MAGIC 
# MAGIC @batch_window_aggregate_feature_view(
# MAGIC     inputs={'transactions_batch': Input(transactions_batch)},
# MAGIC     entities=[user],
# MAGIC     mode='pyspark',
# MAGIC     aggregation_slide_period='1d',
# MAGIC     aggregations=[FeatureAggregation(column='is_evening_weekend', function='sum', time_windows=['30d','60d'])],
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2021, 8, 1)
# MAGIC )
# MAGIC def user_weekend_evening_txn_counts_inline(transactions_batch):
# MAGIC     return is_evening_weekend_inline(is_weekend_inline(is_evening_inline(transactions_batch, "timestamp"), "timestamp")) \
# MAGIC         .withColumnRenamed("nameorig", "user_id") \
# MAGIC         .select('user_id', 'is_evening_weekend', 'timestamp')
# MAGIC </pre>
# MAGIC 
# MAGIC <img src='https://www.doyouevendata.com/wp-content/uploads/2022/01/Screen-Shot-2022-01-02-at-5.56.15-PM.png' width=80%>

# COMMAND ----------

fv = ws.get_feature_view('user_weekend_evening_txn_counts_inline')

# Get a range of feature data - from_source=True required if data not materialized
feature_df = fv.get_historical_features(start_time=datetime(2021, 7, 1), from_source=True)
display(feature_df.to_spark().limit(10))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


