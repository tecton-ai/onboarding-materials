# Databricks notebook source
# MAGIC %md
# MAGIC # Aggregations on Streaming Data
# MAGIC
# MAGIC In this tutorial we'll explore using Tecton to create a feature with the aggregation framework.  We'll cover:
# MAGIC * How to create a streaming aggregate feature with Tecton on Spark natively connecting to a streaming data source (Kafka/Kinesis)
# MAGIC * How to create a streaming aggregate feature using the Tecton managed Ingest API for stream events

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
# MAGIC Tecton has a very powerful framework for creating [time window aggregate features](https://docs.tecton.ai/docs/defining-features/feature-views/time-window-aggregation-functions-reference_).  Note these can be used for both batch feature views as well as streaming feature views.  Typically Batch Feature Views are used for creating aggregates at daily or larger lengths, such as 1 day sales, 3 day sales, 7 day average transaction amount, last 10 stores visited in the past 2 weeks, and so on.  Streaming Feature Views are used for hourly or sub-hour aggregations, with control over how fresh the aggregations need to be as well.  For example, total sales in the past 30 minutes as a declared feature, with freshness incorporating events as recently as sales that occurred 5 minutes ago, 1 minute ago, or 1 second ago - essentially a declaration of how fresh the metric needs to be available/how quickly streaming events should be processed.  Polled and processed every 5 minutes, or processed as soon as an event arrives?
# MAGIC
# MAGIC Tecton additionally offers two ways to interact with streaming data when using Tecton on a Spark backend.  One method is for Tecton to orchestrate Spark Streaming resources that natively listen to a kafka or kinesis topic.  Events are _pulled_ and processed from this stream.  The second method is using Tecton's [Ingest API](https://docs.tecton.ai/using-the-ingestion-api) - under which your own code plucks events off the stream (or from anywhere) with data _pushed_ in to Tecton.  This data can further undergo transformations as well as be used in Tecton aggregations.  Examples of both will be shown below.
# MAGIC
# MAGIC First, let's take a look at some streaming data.  There's already a streaming data source we can leverage to take a look.

# COMMAND ----------

# connect to the prod workspace
ws = tecton.get_workspace('prod')
ws.list_data_sources() 

# COMMAND ----------

transactions_stream = ws.get_data_source('transactions_stream')
transactions_stream.start_stream_preview("temp_table")

# COMMAND ----------

# MAGIC %md
# MAGIC Please note that the **start_stream_preview** command should only be used in a testing environment for short periods of time. The temp_table will grow inifinitely large (memory and time permitting).  Cancel the above command after completing this notebook!
# MAGIC
# MAGIC For additional methods that can be run on stream data source please visit our docs: [StreamSource](https://docs.tecton.ai/api-reference/0.6/stubs/tecton.StreamSource)
# MAGIC
# MAGIC Querying the newly created **temp_table**:

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from temp_table limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC The features we will construct will be the count of transactions larger than $50 in the past 15 minutes, 30 minutes, and 1 hour.  Let's also say we're ok with features that are fresh as of 5 minute intervals, meaning that at any given time, retrieval from the online store would yield a value that was accurate as of the last second to up to 5 minutes, depending on when we retrieved the features.  We can first test some transformation logic to determine whether the transaction is over $50 or not.  We'll declare a stream feature in the notebook and try it out.

# COMMAND ----------

from tecton import stream_feature_view, FilteredSource, Aggregation, StreamProcessingMode

# we already have the data source from the tecton server;
# get the entity from it as well
user = ws.get_entity('fraud_user')

# now declare a test transformation feature
# we're just going to see if it is greater tha $50
# stream_processing_mode here is set to continuous, meaning process the vents as they come in
@stream_feature_view(
    source=FilteredSource(transactions_stream),
    entities=[user],
    mode='spark_sql',
    stream_processing_mode=StreamProcessingMode.CONTINUOUS,
    ttl=timedelta(hours=1),
)
def txn_greater_than_50_test(transactions):
    return f'''
        SELECT
            user_id,
            case when amt > 50 then 1 else 0 end as txn_greater_than_50,
            timestamp
        FROM
            {transactions}
        '''

txn_greater_than_50_test.validate() 

# COMMAND ----------

# MAGIC %md
# MAGIC We can test this transformation on events coming through the stream by writing the output of this feature to a temp table as well.  Note the SQL to show the results will not show results being populated until the following cell is running and events are being populated to the table.  Re-run the SQL cell if results are empty.

# COMMAND ----------

txn_greater_than_50_test.run_stream(output_temp_table="txn_greater_than_50_test_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from txn_greater_than_50_test_table limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before we proceed...
# MAGIC Cancel the above running cells that are continaully processing streaming events before we move on.  These are the `transactions_stream.start_stream_preview("temp_table")` and `txn_greater_than_50_test.run_stream` cells.  The remainder of this tutorial won't source data from the live kinesis stream for this data source, but the history of events that are also defined in the `transaction_stream` data source, contained in partitioned parquet files in s3 as can be seen [here](https://github.com/tecton-ai-ext/tecton-sample-repo/blob/main/fraud/data_sources/transactions.py).

# COMMAND ----------

# MAGIC %md
# MAGIC Declare an aggregate SFV.  In this case, we're going to look at the transformed value above and count how frequently it has occurred in the past 15, 30, and 60 minute windows.  Note some changes to the code below; we've introduced the `aggregations` we are interested in.  These each take and input column, one of the [functions](https://docs.tecton.ai/docs/sdk-reference/time-window-aggregation-functions) we support, and the time window over which to run the aggregate.  We can also specify how fresh the features need to be/how frequently to process the stream.  The example below uses 5 minutes; `aggregation_interval=timedelta(minutes=5)` so every wall clock 5 minutes the new events will be processed.  This can be amended, or the line can be commented along with uncommenting `stream_processing_mode=StreamProcessingMode.CONTINUOUS` to process events as soon as they arrive.
# MAGIC
# MAGIC For additional reading on how these aggregations work in practice, we have a great two part blog available here: [[Part 1]](https://www.tecton.ai/blog/real-time-aggregation-features-for-machine-learning-part-1/), [[Part 2]](https://www.tecton.ai/blog/real-time-aggregation-features-for-machine-learning-part-2/).

# COMMAND ----------

from tecton import stream_feature_view, FilteredSource, Aggregation, StreamProcessingMode

# we already have the data source from the tecton server;
# get the entity from it as well
user = ws.get_entity('fraud_user')

# use the logic from the prior transformation test and count transactions
@stream_feature_view(
    source=FilteredSource(transactions_stream),
    entities=[user],
    mode='spark_sql',
    aggregation_interval=timedelta(minutes=5), # pull events off every wall clock 5 minutes for processing
    #stream_processing_mode=StreamProcessingMode.CONTINUOUS, # pull events off stream continuously as they arrive
    aggregations=[
      Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(minutes=15)),
      Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(minutes=30)),
      Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(hours=1))
    ]
)
def txn_greater_than_50_aggs(transactions):
    return f'''
        SELECT
            user_id,
            case when amt > 50 then 1 else 0 end as txn_greater_than_50,
            timestamp
        FROM
            {transactions}
        '''

txn_greater_than_50_aggs.validate() 

# COMMAND ----------

start_time = datetime.utcnow().replace(microsecond=0, second=0, minute=0, hour=0)-timedelta(hours=3)
end_time = datetime.utcnow()

df = txn_greater_than_50_aggs.get_historical_features(start_time=start_time, end_time=end_time)

display(df.to_pandas().head())

# COMMAND ----------

# MAGIC %md
# MAGIC The feature can optionally be put in to the repository and registered to Tecton via `tecton apply`.  Add a new python file to the repo to define our new feature: `fraud/features/stream_features/txn_greater_than_50_aggs.py` and place the following contents in it.  Note some additional values have been specified in the decorator; to materialize data to the offline store but not online; and to backfill the feature to the start of 2023.  
# MAGIC
# MAGIC ```python
# MAGIC from tecton import stream_feature_view, FilteredSource, Aggregation, StreamProcessingMode
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.transactions import transactions_stream
# MAGIC from datetime import datetime, timedelta
# MAGIC
# MAGIC # windowed counts of transactions over $50
# MAGIC @stream_feature_view(
# MAGIC     source=FilteredSource(transactions_stream),
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     aggregation_interval=timedelta(minutes=5), # pull events off every wall clock 5 minutes for processing
# MAGIC     #stream_processing_mode=StreamProcessingMode.CONTINUOUS, # pull events off stream continuously as they arrive
# MAGIC     aggregations=[
# MAGIC       Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(minutes=15)),
# MAGIC       Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(minutes=30)),
# MAGIC       Aggregation(column='txn_greater_than_50', function='count', time_window=timedelta(hours=1))
# MAGIC     ],
# MAGIC     online=False,
# MAGIC     offline=True,
# MAGIC     feature_start_time=datetime(2023, 1, 1),
# MAGIC )
# MAGIC def txn_greater_than_50_aggs(transactions):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             case when amt > 50 then 1 else 0 end as txn_greater_than_50,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {transactions}
# MAGIC         '''
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the Ingest API
# MAGIC Alternatively, rather than pulling events off of Kafka or Kinesis, streaming data can be pushed in to Tecton via the Ingest API.  In this scenario a client outside of Tecton processes the event, then provides the data to Tecton to ingest.  This data can be pushed straight in with no transformations, or it can undergo python transformations.  It can also be used within the Tecton aggregation framework as well.

# COMMAND ----------

curl -X POST https://preview.staging.tecton.ai/ingest\
     -H "Authorization: Tecton-key $TECTON_API_KEY_STAGING" -d\
 '{
  "workspace_name": "pushapi",
  "dry_run": true,
  "records": {
    "click_event_source": [
      {
        "record": {
          "timestamp": "2023-01-04T00:25:06Z",
          "content_keyword": "apple",
          "clicked": 1
        }
      },
      {
        "record": {
          "timestamp": "2022-12-29T00:25:06Z",
          "content_keyword": "pineapple",
          "clicked": 3
        }
      }
    ],
    "user_click_event_source": [
      {
        "record": {
          "timestamp": "2023-01-02T00:25:06Z",
          "user_id": "pooja",
          "clicked": 2
        }
      },
      {
        "record": {
          "timestamp": "2022-12-29T00:25:06Z",
          "user_id": "emma",
          "clicked": 4
        }
      },
      {
        "record": {
          "timestamp": "2022-12-29T00:25:06Z",
          "user_id": "achal",
          "clicked": 4
        }
      }
    ]
  }
}'