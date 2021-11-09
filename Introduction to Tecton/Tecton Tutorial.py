# Databricks notebook source
# MAGIC %md
# MAGIC ## What is Tecton?
# MAGIC 
# MAGIC Tecton is a data platform for operational machine learning. It empowers data scientists and engineers to:
# MAGIC 
# MAGIC 1. Build great features from batch, streaming, and real-time data
# MAGIC 2. Share and re-use features to build better models faster
# MAGIC 3. Deploy and serve features in production with confidence

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Tecton Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 1.1) Attach to a cluster and run the cell below to import the Tecton SDK
# MAGIC 
# MAGIC In the top-left corner of this notebook, select the cluster that has "-notebook-cluster" in its name to attach this notebook to an interactive Spark cluster. You may need to start the cluster if it is not currently running.
# MAGIC 
# MAGIC To run cells in this notebook, click on the cell and press `shift + enter`.

# COMMAND ----------

# Import Tecton and other libraries
import tecton
import pandas as pd
from datetime import datetime, timedelta

# Check Tecton version
tecton.version.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 1.2) Install the Tecton CLI on your local machine
# MAGIC 
# MAGIC Follow [these instructions in the Tecton Docs](https://docs.tecton.ai/v2/setting-up-tecton/02-tecton-cli-setup.html) to set up the Tecton CLI -- you'll need this on your machine to create new features. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 1.3) Clone the Tecton Sample Repository
# MAGIC 
# MAGIC This tutorial will use [a sample repository full of pre-built features and data sources](https://github.com/tecton-ai-ext/tecton-feature-repo). Before you get started, clone this repository to your local machine.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Tecton Development Environment and Key Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we will explore how to work with Tecton and take a look at the sample Tecton Feature Repo which we will use in the following sections.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: 4 Ways to Interact with Tecton
# MAGIC 
# MAGIC There are 4 ways of interacting with Tecton:
# MAGIC 
# MAGIC 1. **The Tecton Feature Repo and CLI:** Data sources, features, and feature sets are all defined as python configuration files in a local "Feature Repo" typically backed by git (such as the one you cloned earlier). These definitions are then applied to a workspace in a Tecton cluster using the CLI command `tecton apply`. This will be one of the most common CLI commands you will use.
# MAGIC 2. **The Tecton Web UI:** The Tecton Web UI is where you can browse, discover, and monitor all of the data sources, features, and more that have been registered with the cluster using `tecton apply`. This is where you can discover other features in your organization that may be helpful for you model or check on the materialization statuses of new features.
# MAGIC 3. **The Tecton SDK:** Tecton's SDK can be used in any EMR or Databricks notebook (like this one!) to fetch data from the Feature Store. This includes things like previewing feature data, testing transformations, and building training data sets. Currently the SDK requires a Spark Context, but soon we will offer support for using the Tecton SDK from a local notebook without Spark.
# MAGIC 4. **The Tecton REST API:** Tecton's REST API is used for fetching the latest feature values in production for model inference. This endpoint typically returns a feature vector in ~5 milliseconds.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1) Understanding the Feature Repo
# MAGIC 
# MAGIC ##### 🔍 1. Examine the Feature Repo structure
# MAGIC 
# MAGIC <pre>
# MAGIC   (tecton_env) $ ls
# MAGIC     ads
# MAGIC     fraud
# MAGIC 
# MAGIC   (tecton_env) $ ls fraud/
# MAGIC     data_sources
# MAGIC     feature_services
# MAGIC     features
# MAGIC     entities.py
# MAGIC     
# MAGIC   (tecton_env) $ ls fraud/features/
# MAGIC     tests
# MAGIC     batch_feature_views
# MAGIC     batch_window_aggregate_feature_views
# MAGIC     stream_feature_views
# MAGIC     stream_window_aggregate_feature_views
# MAGIC     on_demand_feature_views
# MAGIC     feature_tables
# MAGIC </pre>
# MAGIC 
# MAGIC This structure shows an example of how a Feature Repository can be organized. It includes different folders for different use cases (Ad Prediction and Fraud Detection). We have organized the repo according to our best practices, but you can use whatever directory structure you want. There are no restrictions on how files or directories are organized within a Feature Repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: 5 Key Tecton Objects
# MAGIC 
# MAGIC Inside of your Feature Repo and in your Web UI, you will find references to the 5 main Tecton Objects. Each of these will be explored in more detail in further sections.
# MAGIC 
# MAGIC 1. **Data Sources:** Data sources define a connection to a batch, stream, or request data source and are used as inputs to feature pipelines, known as "Feature Views" in Tecton.
# MAGIC 2. **Feature Views:** Feature Views take in data sources as inputs, or in some cases other Feature Views, and define a pipeline of transformations to compute one or more features. Feature Views also provide Tecton with additional information such as metadata and orchestration, serving, and monitoring configurations.
# MAGIC 3. **Transformations:** Each Feature View has a single pipeline of transformations that define the computation of one or more features. Transformations can be modularized and stitched together into a pipeline, or defined inline in a Feature View (more on this later).
# MAGIC 4. **Entities:** An Entity defines what the feature is about. For example, if the feature is a user's credit score, then the entity is a User. In Tecton, every Feature View is associated with one or more entities.
# MAGIC 5. **Feature Services:** A Feature Service represents a set of features that power a model. Typically there is one Feature Service for each version of a model (where the set of features differ between versions). Feature Services provide convenient endpoints for fetching training data through the Tecton SDK or fetching real-time feature vectors from Tecton's REST API.
# MAGIC 
# MAGIC Ultimately, this mental model boils down to `Data Sources`->`Feature Views`->`Feature Services`.
# MAGIC 
# MAGIC Together these objects define the operational data pipelines you need to run machine learning models in production.
# MAGIC 
# MAGIC <img src="https://docs.tecton.ai/v2/assets/fwv3_concept_diagram.png" width="50%" height="50%" style="margin-left:0px;"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⭐️ Example Use Case: Fraud Detection
# MAGIC 
# MAGIC The rest of this onboarding tutorial will focus on a Fraud Detection use case using objects from the `fraud` folder. In this use case, a user is making a transaction in real-time and we need to predict whether or not it is fraudulent to decide if the transaction should go through. We will need our features to be returned at low-latency so that we don't keep the user waiting, and our features to be as fresh as possible to respond to recent information and give accurate predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Fetch Data
# MAGIC 
# MAGIC There are several registered data sources in this cluster such as `users_batch`, `credit_scores_batch`, and `transactions_stream`. These data sources are configured in files under `fraud/data_sources` and define how Tecton connects to your organization's raw data sources.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 3.1) Preview a data source
# MAGIC Let's take a look at data from a data source using Tecton's SDK in the cell below.

# COMMAND ----------

tecton.list_data_sources() # List Data Sources registered with Tecton

# COMMAND ----------

credit_scores_batch = tecton.get_data_source('credit_scores_batch') # swap with any data source name, e.g. transactions_stream, etc.
display(credit_scores_batch.get_dataframe().to_spark().limit(10)) # credit_scores_batch.get_dataframe() returns the underlying data as a Tecton DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see above, Tecton can return data from a registered Data Source as a Spark DataFrame. This DataFrame can be used to explore the data and test potential transformations that you want to use in a Feature View.
# MAGIC 
# MAGIC Note: In the case of a Stream Data Source, Tecton will use a batch log of historical stream events to fetch data with the code above. The batch log of stream events is part of the registration of a Stream Data Source in the Feature Repo.
# MAGIC 
# MAGIC However, Tecton also supports consuming data from a registered Stream Data Source, directly from the stream!
# MAGIC 
# MAGIC #### ✅ Run the cells below to preview data from the stream source.
# MAGIC 
# MAGIC * When you run Cmd 19, you'll need to wait as the stream initializes. Once it says 'temp_table' you can move to the next step.
# MAGIC * After you run Cmd 20 and see some sample results, please go back and cancel Cmd 19 to kill the stream job.

# COMMAND ----------

transactions_stream = tecton.get_data_source('transactions_stream')
transactions_stream.start_stream_preview("temp_table")

# COMMAND ----------

display(spark.sql("SELECT * FROM temp_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 3.2) Preview Feature View data
# MAGIC Let's check out some data from a registered Feature View. You can list Feature Views using the code below and also find them in the Web UI or Feature Repo.

# COMMAND ----------

tecton.list_feature_views() # you can list any Tecton objects such as `tecton.list_data_sources()` and `tecton.list_feature_services()`

# COMMAND ----------

# MAGIC %md
# MAGIC Now you can use the `get_historical_features()` call to get a DataFrame of historical feature values from the Offline Feature Store. `get_historical_features` also allows you to easily filter for a time range of values, or for a specific set of entity keys.
# MAGIC 
# MAGIC `get_historical_features()` is a good way to better understand and explore features from the Feature Store, but it can also be used to test out newly developed features (more on this later).
# MAGIC 
# MAGIC #### ✅ Run the cell below to fetch features from the Offline Feature Store

# COMMAND ----------

fv = tecton.get_feature_view("last_transaction_amount_pyspark") # try replacing with any other materialized Feature View name from above
feature_df = fv.get_historical_features(start_time=(datetime.now() - timedelta(days = 30)), from_source=True) # `end_time` defaults to "now" so this will fetch the last 30 days of data

display(feature_df.to_spark().orderBy("timestamp", ascending=False)) # feature_df is a "Tecton DataFrame" which can also be converted to Pandas using `feature_df.to_pandas()`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Generate Training Data
# MAGIC 
# MAGIC Let's generate some training data using a Tecton Feature Service and a set of training events (called a "spine").
# MAGIC 
# MAGIC As mentioned above, a Feature Service is a Tecton object that represents a set of features for a model and provides endpoints for getting training and serving data. Typically there is one Feature Service for each version of a model (where the set of features differ between versions).
# MAGIC 
# MAGIC The Feature Service we will fetch, `fraud_detection_feature_service`, can be used to power our production Fraud Detection model. In the outputted summary below you will see the list of features used in this Feature Service, coming from various included Feature Views. This can also be seen in the Web UI.
# MAGIC 
# MAGIC #### ✅ Run the cell below to fetch an existing production FeatureService and view a summary of metadata

# COMMAND ----------

fraud_detection_feature_service = tecton.get_feature_service("fraud_detection_feature_service")
fraud_detection_feature_service.summary() # all Tecton objects have a `.summary()` method that can be used to fetch metadata also found in the Web UI

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: Spine Dataframe
# MAGIC 
# MAGIC A "spine" dataframe represents the historical training events for which to join in feature values. This is used as an input to a Feature Service when fetching historical features.
# MAGIC 
# MAGIC A spine must include:
# MAGIC 1. The join keys of all entities for any Feature Views in the Feature Service, as seen above in the Feature Service summary under Offline Join Keys. For example, if your Feature Views include User and Transaction entities, then your spine needs to include user_id and transaction_id columns.
# MAGIC 2. A corresponding event timestamp that represents the point in time to fetch historical feature values. Tecton uses these timestamps to do time-travel on a row-level basis and prevent data leakage.
# MAGIC 3. Any historical request data for each event. This is only needed if the Feature Service uses a Feature View that depends on a Request Data Source where some raw data is passed in with the request to a Feature Service. Here the amount for the transaction being scored is passed in as request data.
# MAGIC 
# MAGIC A spine typically also includes a label column for the training events. Tecton will preserve additional columns like this in the spine but does not require the label for generating training data.
# MAGIC 
# MAGIC Our spine DataFrame below represents historical training events of user transactions that were labeled as fraudulent or not fraudulent.
# MAGIC 
# MAGIC #### ✅ Run the cell below to generate a spine

# COMMAND ----------

from pyspark.sql import functions as F

# 1. Fetching a Spark DataFrame of historical labeled transactions
# 2. Renaming columns to match the expected join keys for the Feature Service
# 3. Selecting the join keys, request data, event timestamp, and label
training_events = tecton.get_data_source("transactions_batch").get_dataframe().to_spark() \
                        .withColumnRenamed("nameorig", "user_id") \
                        .select("user_id", "amount", "timestamp", "isfraud") \
                        .filter("partition_0 == '2021'") \
                        .filter("partition_1 == '10'") \
                        .filter("partition_2 == '08'") \
                        .orderBy("timestamp", ascending=False) \
                        .limit(100).cache()
display(training_events)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅ Now run the cell below to use the generated spine and the retrieved Feature Service to generate training data

# COMMAND ----------

training_data = fraud_detection_feature_service.get_historical_features(spine=training_events, timestamp_key="timestamp").to_spark()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Get Feature Vectors for Predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 5.1) Get current feature vectors for real-time predictions
# MAGIC 
# MAGIC Now let's use this same Feature Service to fetch feature vectors in real time at low latency by running the cell below.
# MAGIC 
# MAGIC In production, we will use the REST API to fetch real-time feature values, but the SDK also provides a convenience method for testing purposes. The SDK method uses the REST API internally.
# MAGIC 
# MAGIC To request a feature vector, we need to pass in:
# MAGIC 
# MAGIC 1. The join keys for which we want to retrieve features. In this example, we're providing user_id for the transaction.
# MAGIC 1. The request_data with any request-time data required for our Feature Views. In this example, we need to provide the amount of the transaction.

# COMMAND ----------

from pprint import pprint

keys = {'user_id': 'C439568473'}
request_data = {'amount': 1122.14}

response = fraud_detection_feature_service.get_online_features(keys, request_data=request_data)
pprint(response.to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's also try using the REST API to query real-time feature values.
# MAGIC 
# MAGIC #### ✅Generate an API key in your CLI by running:
# MAGIC 
# MAGIC <pre>
# MAGIC $ tecton create-api-key
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅ Now export that key to an environment variable called TECTON_API_KEY
# MAGIC 
# MAGIC <pre>
# MAGIC $ export TECTON_API_KEY=[your_key]
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅Now query the REST API using the cURL command below! Make sure to fill in your correct cluster name in the first line.
# MAGIC 
# MAGIC > 💡**Tip:** You can find an example cURL for every Feature Service in the Web UI
# MAGIC 
# MAGIC <pre>
# MAGIC curl -X POST https://[your-cluster-name].tecton.ai/api/v1/feature-service/get-features\
# MAGIC      -H "Authorization: Tecton-key $TECTON_API_KEY" -d\
# MAGIC '{
# MAGIC   "params": {
# MAGIC     "feature_service_name": "fraud_detection_feature_service",
# MAGIC     "join_key_map": {
# MAGIC       "user_id": "C439568473"
# MAGIC     },
# MAGIC     "request_context_map": {
# MAGIC       "amount": 1122.14
# MAGIC     }
# MAGIC   }
# MAGIC }'
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Create a new feature

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we will create a new feature and a new experimental Feature Service to test that feature in a training dataset. First we will start off by creating our own Tecton Workspace for our personal experimentation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: Tecton Workspaces
# MAGIC 
# MAGIC Workspaces are like a sandbox environment that can be used for experimenting with a Feature Repo without affecting the production environment. Changes made in one workspace will have no affect on other Workspaces.
# MAGIC 
# MAGIC By default, non-"prod" workspaces do not have access to materialization and storage resources. Instead, transformations can be ran ad-hoc using your notebook's attached Spark cluster.
# MAGIC 
# MAGIC This ad-hoc computation functionality can be used in any workspace and allows you to easily test features without needing to backfill and materialize data to the Feature Store.
# MAGIC 
# MAGIC New workspaces with full materialization and storage resources can be created with the addition of the _--live_ flag during create time as shown in the below CLI command. This can be useful for creating staging environments or testing features online before pushing changes to prod.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.1) Create your own Tecton Workspace
# MAGIC 
# MAGIC Workspaces are created using the Tecton CLI. Let's create your own personal workspace and apply your local feature repo changes to it.
# MAGIC 
# MAGIC Creating a new Workspace has no effect on the structure of your repo and can be executed at any time.
# MAGIC 
# MAGIC To get started, create a workspace by running `tecton workspace create YOUR_NAME`.  Your newly created workspace will be empty at first - you'll need to apply a feature repository to populate it with features.
# MAGIC 
# MAGIC <pre>
# MAGIC   $ tecton workspace create YOUR_NAME
# MAGIC   Created workspace "YOUR_NAME".
# MAGIC   Switched to workspace "YOUR_NAME".
# MAGIC 
# MAGIC   You're now on a new, empty workspace. Workspaces isolate their state,
# MAGIC   so if you run "tecton plan" Tecton will not see any existing state
# MAGIC   for this configuration.
# MAGIC </pre>
# MAGIC 
# MAGIC > 💡**Tip:** For a complete list of workspace commands, simply run `tecton workspace -h`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: Applying a local Feature Repo to a Workspace
# MAGIC 
# MAGIC Tecton manages all data sources, features, and services in a remote registry for each workspace. When new Tecton object definitions from the local Feature Repository are applied to a workspace they become available for use through the Tecton Python SDK (used in this notebook) and discoverable via the Tecton Web UI (by changing the selected workspace up top).
# MAGIC 
# MAGIC When using Tecton, you'll use the command `tecton apply` to take the changes in your local Feature Repo and apply them to the workspace. `tecton apply` always outputs a plan which describes the changes that will be safely made to move your remote workspace from its current state to the new desired state defined locally. To see just the plan by itself, you can use `tecton plan`.
# MAGIC 
# MAGIC > 💡**Tip:** You can switch which workspace you are applying changes to by running `tecton workspace select WORKSPACE_NAME`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Concept: Feature Development with Workspaces and Git
# MAGIC 
# MAGIC Workspaces are designed to work naturally with Git. In practice, a typical feature development flow with Git looks like this:
# MAGIC 
# MAGIC 1. Create a new empty workspace using `tecton workspace create WORKSPACE_NAME`
# MAGIC 2. Create a new Git branch
# MAGIC 3. Apply the existing Feature Repo to the empty workspace using `tecton apply`
# MAGIC 4. Do exploratory data analysis and test transformations in a notebook
# MAGIC 4. Add a new Feature View to the local Feature Repo, apply changes using `tecton apply`, and commit them to Git.
# MAGIC 5. Test changes in your notebook
# MAGIC 6. Merge your changes to the main branch
# MAGIC 7. Switch to the "prod" workspace with `tecton workspace select prod` and run `tecton apply`. Often this step is handled by a CI/CD pipeline.
# MAGIC 8. (Optional) Delete your experimental workspace
# MAGIC 
# MAGIC 🎉**Coming Soon:** Tecton will offer a direct integration with Git so that the workspace management steps can be automatically handled for you!
# MAGIC 
# MAGIC Using Git is totally optional for this tutorial.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.2) Run `tecton apply` to apply the local Feature Repo to your new Workspace
# MAGIC 
# MAGIC Because you are working in a fresh new Workspace, it does not currently have anything registered. Therefore, you will first want to apply the existing local feature repository to the workspace by running `tecton apply`. If everything is set up correctly, you should see something like this:
# MAGIC 
# MAGIC <pre>
# MAGIC ➜  tecton-feature-repo git:(main) tecton apply
# MAGIC Using workspace "matt" on cluster https://staging.tecton.ai
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
# MAGIC </pre>
# MAGIC 
# MAGIC Take note of the workspace you are applying to to make sure it is correct. Then go ahead and apply the plan with `y`.
# MAGIC 
# MAGIC > 💡 **Tip:** You can always compare your local Feature Repo to the remote Feature Registry by running `tecton plan`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.3) Brainstorming a New Feature
# MAGIC 
# MAGIC Frequently the easiest way to get started building a new feature is by drafting some transformation logic directly in a notebook. Let's do that here, building a feature that checks if the user has great credit or not (over 740).

# COMMAND ----------

credit_scores_batch = tecton.get_data_source("credit_scores_batch").get_dataframe().to_spark()
credit_scores_batch.createOrReplaceTempView("credit_scores")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     user_id,
# MAGIC     IF (credit_score > 740, 1, 0) as user_has_great_credit,
# MAGIC     timestamp
# MAGIC FROM
# MAGIC     credit_scores

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4) Create a new Feature View using this transformation in the Feature Repo
# MAGIC 
# MAGIC Now that we've tested out our transformation logic, let's turn that logic into a Feature View so that we can start generating training data.
# MAGIC 
# MAGIC ✅ Navigate to `fraud/features/batch_feature_views/user_has_great_credit.py` and uncomment the Feature View code shown below:
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
# MAGIC We can define a Feature View for this transformation using the decorator class shown above.
# MAGIC 
# MAGIC The `inputs`, `entities`, `mode`, `batch_schedule`, and `ttl` parameters are the only ones required to register the feature and test it using the ad-hoc compute in a notebook. The remaining parameters configure materialization and backfilling information, as well as metadata.
# MAGIC 
# MAGIC The materialization jobs however won't begin until we apply this definition to the "prod" workspace which has materialization and storage resources.
# MAGIC 
# MAGIC Lastly, the `mode='spark_sql'` parameter tells Tecton that we are defining this Feature View pipeline with a single inline Spark SQL transformation. The mode can also be `pyspark` for PySpark transformation code or `pipeline` to define a pipeline of transformations as shown below.
# MAGIC 
# MAGIC The inline transformation is shorthand for:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5) Create a new Fraud Detection Feature Service variant with this new feature
# MAGIC 
# MAGIC If we want to generate training data that includes our new feature, we'll need to include it in a Feature Service.
# MAGIC 
# MAGIC ✅ Navigate to `fraud/feature_services/fraud_detection.py` and uncomment the new Feature Service code shown below, as well as the import statement for the new Feature View at the top of the file:
# MAGIC 
# MAGIC <pre>
# MAGIC from fraud.features.batch_feature_views.user_has_great_credit import user_has_great_credit
# MAGIC 
# MAGIC fraud_detection_feature_service_v2 = FeatureService(
# MAGIC     name='fraud_detection_feature_service:v2',
# MAGIC     description='A FeatureService providing features for a model that predicts if a transaction is fraudulent.',
# MAGIC     family='fraud',
# MAGIC     tags={'release': 'production'},
# MAGIC     features=[
# MAGIC         user_has_great_credit, # New feature
# MAGIC         last_transaction_amount_sql,
# MAGIC         transaction_amount_is_high,
# MAGIC         transaction_amount_is_higher_than_average,
# MAGIC         user_transaction_amount_metrics,
# MAGIC         user_transaction_counts,
# MAGIC         user_distinct_merchant_transaction_count_30d
# MAGIC     ]
# MAGIC )
# MAGIC </pre>
# MAGIC 
# MAGIC Adding the `:v2` appendix to the Feature Service name tells Tecton to treat this as a "variant" of the `fraud_detection_feature_service`. Tecton will group these Feature Service variants together in the Web UI. Variants can be used to manage new versions of Feature Views and Feature Services and handle migrations.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.6) Apply the Feature View and Feature Service to your workspace using the command `tecton apply`.
# MAGIC <pre>
# MAGIC ➜  tecton-sample-repo git:(main) ✗ tecton apply
# MAGIC Using workspace "prod" on cluster https://app.tecton.ai
# MAGIC ✅ Imported 45 Python modules from the feature repository
# MAGIC ✅ Running Tests: Tests passed!
# MAGIC ✅ Collecting local feature declarations
# MAGIC ✅ Performing server-side validation of feature declarations
# MAGIC  ↓↓↓↓↓↓↓↓↓↓↓↓ Plan Start ↓↓↓↓↓↓↓↓↓↓
# MAGIC 
# MAGIC   + Create Transformation
# MAGIC     name:            user_has_great_credit
# MAGIC     description:     Whether the user has a great credit score (over 740).
# MAGIC 
# MAGIC   + Create BatchFeatureView
# MAGIC     name:            user_has_great_credit
# MAGIC     description:     Whether the user has a great credit score (over 740).
# MAGIC 
# MAGIC   + Create FeatureService
# MAGIC     name:            fraud_detection_feature_service:v2
# MAGIC     description:     A FeatureService providing features for a model that predicts if a transaction is fraudulent.
# MAGIC 
# MAGIC  ↑↑↑↑↑↑↑↑↑↑↑↑ Plan End ↑↑↑↑↑↑↑↑↑↑↑↑
# MAGIC Note: Updates to Feature Services may take up to 60 seconds to be propagated to the real-time feature-serving endpoint.
# MAGIC Are you sure you want to apply this plan? [y/N]> y
# MAGIC 🎉 all done!
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.7) Test your new Feature View
# MAGIC The cell below fetches our new Feature View and then dry runs the transformation pipeline over a specified time range. If no `feature_start_time` is passed in, it will default to `feature_end_time` - `batch_schedule` to mimic a single materialization run. If neither `feature_start_time` nor `feature_end_time` are passed, `.run()` will use the most recent `batch_schedule` time.

# COMMAND ----------

user_has_great_credit = tecton.get_workspace("YOUR_NAME").get_feature_view("user_has_great_credit") # objects fetched from non-prod workspaces need to first specify the workspace with `.get_workspace("workspace_name")`
feature_df = user_has_great_credit.run(feature_start_time=datetime(2021,7,1), feature_end_time=datetime(2021,8,1))

display(feature_df.to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.8) Test your new Feature Service
# MAGIC 
# MAGIC We can reuse the same spine we previously generated, but with our new Feature Service.

# COMMAND ----------

fraud_detection_feature_service_v2 = tecton.get_workspace("YOUR_NAME").get_feature_service("fraud_detection_feature_service:v2")
training_data = fraud_detection_feature_service_v2.get_historical_features(spine=training_events, timestamp_key="timestamp", from_source=True).to_spark()
display(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ 6.9) Switch to the "prod" workspace and apply your changes
# MAGIC 
# MAGIC Your final step is simply to switch back to the "prod" workspace and apply your changes by running `tecton workspace select prod` and then `tecton apply`.
# MAGIC 
# MAGIC As noted above, it is common to have CI/CD run the `tecton apply` to production upon merged changes to master.
# MAGIC 
# MAGIC Now that you have applied your Feature View to the "prod" workspace with materialization resources you can navigate to the Feature View's Materialization tab in the Web UI and see that Tecton is actively backfilling feature data to the online and offline store!
# MAGIC 
# MAGIC Once the materialization jobs have finished, you will be able to cURL your new Feature Service for online feature values following the previous examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC In this walkthrough we:
# MAGIC 1. Learned Tecton key concepts
# MAGIC 2. Set up a Tecton development environment
# MAGIC 3. Fetched data from the Feature Store
# MAGIC 4. Generated training data and inference data from a set of existing features
# MAGIC 5. Defined and tested a new feature and feature set and deployed it to production
# MAGIC 
# MAGIC Now you are ready to define, serve, and monitor feature pipelines and services using Tecton!

# COMMAND ----------

# MAGIC %md
# MAGIC ## What's next?
# MAGIC 
# MAGIC Try exploring the other feature examples in the Feature Repo and our documentation and creating more of your own!
# MAGIC 
# MAGIC In this tutorial we only scratched the surface of the powerful types of feature pipelines that can be created in Tecton. We recommend also checking out:
# MAGIC 
# MAGIC 1. Highly efficient streaming (or batch) aggregation features (e.g. `fraud/features/stream_window_aggregate_feature_views/user_transaction_amount_metrics`)
# MAGIC 2. On-demand features that run at request time and can leverage request data and other Feature Views (e.g. `fraud/features/on_demand_feature_views/transaction_amount_is_higher_than_average.py`)
# MAGIC 
# MAGIC ### Tecton Documentation
# MAGIC - <a href="https://docs.tecton.ai" target="_blank">Tecton Docs</a>
# MAGIC - <a href="https://s3-us-west-2.amazonaws.com/tecton.ai.public/documentation/tecton-py/index.html" target="_blank">Tecton Python SDK Docs</a>
# MAGIC 
# MAGIC ### Have Questions?
# MAGIC Hit us up in Slack! We have a public Slack workspace that you can join here:
# MAGIC 
# MAGIC <a href="https://slack.feast.dev/">https://slack.feast.dev/</a>
# MAGIC 
# MAGIC You can send your questions to the #tecton-questions channel.
# MAGIC 
# MAGIC We look forward to seeing what you create!
