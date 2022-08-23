# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Building Features with Tecton
# MAGIC 
# MAGIC In this tutorial we'll cover how you can use Tecton to build features for machine learning.  We'll cover:
# MAGIC * How to register features with Tecton
# MAGIC * How features are written in Tecton
# MAGIC * How to use Tecton Aggregations to do easy window aggregations

# COMMAND ----------

# MAGIC %md
# MAGIC ### ❓ Before we start -- Tecton Workspaces
# MAGIC 
# MAGIC [Workspaces](https://docs.tecton.ai/overviews/workspaces.html) are like a sandbox environment that can be used for experimenting with a Feature Repo without affecting the production environment. Changes made in one workspace will have no affect on other Workspaces.
# MAGIC 
# MAGIC By default, new "development" workspaces do not have access to materialization and storage resources. Instead, transformations can be run ad-hoc in your Snowflake Warehouse. This means that the Tecton SDK builds a query that reads directly from your raw data tables, and executes it in your Snowflake Warehouse.
# MAGIC 
# MAGIC This ad-hoc computation functionality can be used in any workspace and allows you to easily test features without needing to backfill and materialize data to the Feature Store.
# MAGIC 
# MAGIC New workspaces with full materialization and storage resources can be created with the addition of the _--live_ flag during create time in the below CLI command. This can be useful for creating staging environments for testing features online before pushing changes to prod, or for creating isolation between different teams.
# MAGIC 
# MAGIC **In this tutorial, we'll create a new workspace to ensure our changes don't effect other's workloads**

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Create your own Tecton Workspace
# MAGIC 
# MAGIC Workspaces are created using the Tecton CLI. Let's make one now:
# MAGIC 
# MAGIC Create a workspace by running `tecton workspace create YOUR_NAME`.
# MAGIC 
# MAGIC ```
# MAGIC $ tecton workspace create YOUR_NAME
# MAGIC Created workspace "YOUR_NAME".
# MAGIC Switched to workspace "YOUR_NAME".
# MAGIC 
# MAGIC You're now on a new, empty workspace. Workspaces isolate their state,
# MAGIC so if you run "tecton plan" Tecton will not see any existing state
# MAGIC for this configuration.
# MAGIC ```
# MAGIC 
# MAGIC > 💡**Tip:** For a complete list of workspace commands, simply run `tecton workspace -h`

# COMMAND ----------

# MAGIC %md
# MAGIC ### ❓ Before we start -- Tecton Feature Repos
# MAGIC 
# MAGIC In Tecton, [features are declared as code](https://docs.tecton.ai/examples/managing-feature-repos.html), in a **Tecton Feature Repository**. When your team uses Tecton, in practice you'll be collaborating on a code repository that defines all of the features that you expect Tecton to manage.
# MAGIC 
# MAGIC That means before we build a new feature, we'll need to clone the code repository that your team will use to collaborate on features.  **In this tutorial, we'll clone a pre-populated feature repository to use as a starting point**

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Clone the Sample Feature Repo
# MAGIC 
# MAGIC The [sample feature repository for this demo can be found here](https://github.com/tecton-ai-ext/tecton-snowflake-feature-repo) -- if you already checked out this git repository to get a copy of this tutorial, you should already have the important files downloaded.  If not, clone the sample repository -- in the next steps you'll be editing files in that repo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Apply the Sample Feature Repo
# MAGIC 
# MAGIC To register a local feature repository with Tecton, [you'll use the Tecton CLI.](https://docs.tecton.ai/examples/managing-feature-repos.html) Since you are working in a new Workspace, it does not currently have anything registered, so your first time adding features should be simple.
# MAGIC 
# MAGIC Navigate to the feature repository's directory in the command line:
# MAGIC ```
# MAGIC cd feature_repo
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC Then run the following command to register your feature definitions with Tecton:
# MAGIC ```
# MAGIC tecton apply
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC Take note of the workspace you are applying to to make sure it is correct. Then go ahead and apply the plan with `y`.
# MAGIC 
# MAGIC > 💡 **Tip:** You can always compare your local Feature Repo to the remote Feature Registry before applying it by running `tecton plan`.

# COMMAND ----------

# MAGIC %md
# MAGIC # Building your first feature
# MAGIC 
# MAGIC On to the fun part, let's build a feature in Tecton.
# MAGIC 
# MAGIC ### Setup
# MAGIC 
# MAGIC ✅ Run the cell below.

# COMMAND ----------

import tecton
import pandas as pd
from datetime import datetime, timedelta
from pprint import pprint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constructing a Feature
# MAGIC Let's start by building a simple feature -- **the amount of the last transaction a user made**. First, let's run a query against the raw data in Snowflake (feel free to run this yourself in a Snowflake worksheet as well).

# COMMAND ----------

# Preview the data directly
feature_query = '''
SELECT 
    user_id,
    amt,
    to_timestamp(timestamp) as timestamp
FROM 
    demo_fraud_v2.transactions 
'''
feature_df = spark.sql(feature_query)
display(feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In Tecton, a feature has **three key components**:
# MAGIC 1. A set of keys that specify who or what the feature is describing (associated with an [Entity](https://docs.tecton.ai/overviews/framework/entities.html)). In the above example, the key is `USER_ID`, meaning this feature is describing a property about a user.
# MAGIC 2. One or more feature values -- the stuff that's going to eventually get passed into a model.  In the above example, the feature is `AMT`, the amount of the transaction.
# MAGIC 3. A timestamp for the feature value. In the above example, the timestamp is `TIMESTAMP`, signifying that the feature is valid as of the moment of the transaction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining a Feature to Tecton
# MAGIC Moving from your Snowflake query to a Tecton feature is very simple, you'll simply wrap the SQL query in a Tecton python decorator.  Here's what it looks like in practice:
# MAGIC 
# MAGIC ```python
# MAGIC @batch_feature_view(
# MAGIC     sources=[transactions_batch],
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     batch_schedule=timedelta(days=1),
# MAGIC     ttl=timedelta(days=30),
# MAGIC     feature_start_time=datetime(2022, 5, 1),
# MAGIC     description='Last user transaction amount (batch calculated)'
# MAGIC )
# MAGIC def user_last_transaction_amount(transactions):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             amt,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {transactions}
# MAGIC         '''
# MAGIC ```
# MAGIC 
# MAGIC ✅  To add this feature to Tecton, simply add it to a new file in your Tecton Feature Repository.
# MAGIC 
# MAGIC ✅  Once you save your new feature, run `tecton apply` to publish it to Tecton.
# MAGIC 
# MAGIC Currently this feature has online materialization disabled. If the `online=True` flag is enabled when the Feature View is applied, Tecton will automatically backfill feature data to the online store from the specified `feature_start_time` until `now` (the time at which you apply the repository). Afterwards, the feature store gets refreshed at every `batch_schedule` interval going forward.
# MAGIC 
# MAGIC As shown in the last tutorial, we can test run this new Feature view using the `.run()` function below.

# COMMAND ----------

YOUR_WORKSPACE_NAME ="PUT_YOUR_WORKSPACE_NAME_HERE"

ws = tecton.get_workspace(YOUR_WORKSPACE_NAME) # replace with your workspace name
fv = ws.get_feature_view('user_last_transaction_amount')

start_time = datetime.utcnow()-timedelta(days=10)
end_time = datetime.utcnow()

fv.run(start_time=start_time, end_time=end_time).to_pandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Tecton time-windowed aggregations
# MAGIC Sliding time-windowed aggregations are common ML features for event data, but defining them in a view can be error-prone and inefficient.
# MAGIC 
# MAGIC Tecton provides built-in implementations of common time-windowed aggregations that simplify transformation logic and ensure correct feature value computation. Additionally, Tecton optimizes the compute and storage of these aggregations to maximize efficiency.
# MAGIC 
# MAGIC For these reasons, we recommend using Tecton’s built-in aggregations whenever possible.
# MAGIC 
# MAGIC Time-windowed aggregations can be specified in the Batch Feature View decorator using the `aggregations` and `aggregation_interval` parameters.
# MAGIC 
# MAGIC Tecton expects the provided SQL query to select the raw events (with timestamps) to be aggregated.
# MAGIC 
# MAGIC ```python
# MAGIC @batch_feature_view(
# MAGIC     sources=[transactions_batch],
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     feature_start_time=datetime(2022, 5, 1),
# MAGIC     description='Max transaction amounts for the user in various time windows',
# MAGIC     aggregation_interval=timedelta(days=1),
# MAGIC     aggregations=[Aggregation(column='amt', function='max', time_window=timedelta(days=7))],
# MAGIC )
# MAGIC def user_max_transactions(transactions):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             amt,
# MAGIC             timestamp
# MAGIC         FROM
# MAGIC             {transactions}
# MAGIC         '''
# MAGIC ```
# MAGIC 
# MAGIC ✅  To add this feature to Tecton, simply add it to a new file in your Tecton Feature Repository.
# MAGIC 
# MAGIC ✅  Once you save your new feature, run `tecton apply` to publish it to Tecton.
# MAGIC 
# MAGIC Now we can test this feature below.

# COMMAND ----------

fv = ws.get_feature_view('user_max_transactions')

start_time = datetime.utcnow()-timedelta(days=30)
end_time = datetime.utcnow()

fv.run(start_time=start_time, end_time=end_time).to_pandas().fillna(0).head()

# COMMAND ----------

# MAGIC %md
# MAGIC If you want to add these features to your feature set for your model, simply extend the list of Feature Views in your [Feature Service](feature_repo/feature_services/fraud_detection.py).
