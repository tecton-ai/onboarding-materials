# Databricks notebook source
# MAGIC %md
# MAGIC # Creating A Feature
# MAGIC
# MAGIC In this tutorial we'll use Tecton to create a new feature.  We'll cover:
# MAGIC * How to use Notebook Driven Development (NDD) to declare and test out a new feature in local code
# MAGIC * How to create a workspace in Tecton
# MAGIC * How to apply a feature to the Tecton server in a workspace

# COMMAND ----------

# import tecton and other libraries
import os
import tecton
import pandas as pd
from datetime import datetime, timedelta

# check tecton version
tecton.version.summary()

# COMMAND ----------

# connect to the prod workspace
ws = tecton.get_workspace('prod')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.) Declare and test out a new feature
# MAGIC
# MAGIC #### 1.1) Develop feature logic
# MAGIC
# MAGIC The new feature we'll define will be a simple transformational feature.  The business logic will examine the first digit of a credit card number and map it to an issuing bank.  The minimal set of fields required to create a feature will consist of a key to look the feature up by, features themselves, and then a timestamp representing when they became available.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   user_id,
# MAGIC   CASE SUBSTRING(CAST(cc_num AS STRING), 0, 1)
# MAGIC     WHEN '4' THEN 'Visa'
# MAGIC     WHEN '5' THEN 'MasterCard'
# MAGIC     WHEN '6' THEN 'Discover'
# MAGIC     ELSE 'other'
# MAGIC   END as credit_card_issuer,
# MAGIC   signup_timestamp
# MAGIC   FROM demo_fraud_v2.customers limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2) Leverage existing registered Tecton objects
# MAGIC
# MAGIC Our logic looks good, let's create a new feature now.  We'll first create a feature within the scope of this notebook, leveraging some of the existing Tecton objects - notably an existing data source and entity.  We can see the existing objects in this workspace. We will use the users_batch data source and fraud_user entity.

# COMMAND ----------

ws.list_data_sources()

# COMMAND ----------

ws.list_entities()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3) Create notebook scoped Batch Feature View (BFV)
# MAGIC
# MAGIC Leveraging the above logic and Tecton objects, we can create a Tecton BFV scoped to this tutorial for testing and evaluation. Below a Tecton BFV is declared inline, and will use the Tecton registered users_batch data source and fraud_user entity objects.  The declared BFV is then validated by the SDK for use.  Note Tecton features are defined by python functions with python decorators.  The `mode` is the actual compute and language the feature will run as.  The `sources`, `entities`, `mode`, `batch_schedule`, and `ttl` parameters are the only ones required to register the feature and test it using the ad-hoc compute in a notebook. The remaining parameters configure materialization and backfilling information, as well as metadata.

# COMMAND ----------

from tecton import batch_feature_view

users = ws.get_data_source('users_batch')
user = ws.get_entity('fraud_user')

@batch_feature_view(
    sources=[users],
    entities=[user],
    mode='spark_sql',
    batch_schedule=timedelta(days=1),
    ttl=timedelta(days=365 * 30),
    timestamp_field='signup_timestamp'
)
def cc_issuer(users):
  return f'''
    SELECT
      user_id,
      CASE SUBSTRING(CAST(cc_num AS STRING), 0, 1)
        WHEN '4' THEN 'Visa'
        WHEN '5' THEN 'MasterCard'
        WHEN '6' THEN 'Discover'
        ELSE 'other'
      END as credit_card_issuer,
      signup_timestamp
      from {users}
  '''

cc_issuer.validate() # validate the above BFV declaration looks good to use and register for use within notebook

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4) Test out the feature
# MAGIC
# MAGIC The feature can be tested with features calculated against the data source.  Note the effective timestamp of the feature is provided as well; this is the time the value would be expected to actually appear in Tecton.  As this feature was defined on a daily batch schedule, the effective timestamp is the subsequent day after the raw data was created (signup_timestamp).

# COMMAND ----------

start_time = datetime(2017, 1, 1)
end_time = datetime.utcnow()

df = cc_issuer.get_historical_features(start_time=start_time, end_time=end_time)

display(df.to_pandas().head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5) Test out the feature in a feature service
# MAGIC
# MAGIC We can similarly declare a feature service.  This can be done from scratch or by extending an existing feature service; we'll leverage the one from the prior tutorial and add the above feature to it.  A training data set will then be created with it.  This data set could be used to assess lift of the new feature by a model.

# COMMAND ----------

existing_fs = ws.get_feature_service('fraud_detection_feature_service')

test_fs = tecton.FeatureService(
  name='test_fs',
  features = existing_fs.features + [cc_issuer]
)

test_fs.validate()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we'll use a simple spine created on the fly with a dataframe and pass it in to retrieve some features as an example.  We'll assume this feature has some lift, and for our next step, we'll register it within Tecton itself!

# COMMAND ----------

spine = pd.DataFrame.from_records([{
    "user_id": "user_934384811883",
    "merchant": "fraud_Rempel Inc",
    "amt": 135.72,
    "timestamp": datetime(2023, 3, 29, 0),
},
{
    "user_id": "user_402539845901",
    "merchant": "fraud_Shanahan-Lehner",
    "amt": 227,
    "timestamp": datetime(2023, 3, 29, 0),
}])

df = test_fs.get_historical_features(spine)
display(df.to_spark())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.) Tecton Workspaces
# MAGIC
# MAGIC [Workspaces](https://docs.tecton.ai/overviews/workspaces.html) are environments within a Tecton instance.  These can be used for personal or team development.  They can also be specified as _--live_ or not during their creation.  Live workspaces allow for materialization to offline and online stores; non-live workspaces will not materialize data and are intended for non-production usage.  To use the online store, an environment must be made live.  The offline store can be used in both live and non-live workspaces, but the latter will involve computing features from data source when requested rather than retrieving precomputed values from the offline store.  Changes made in one workspace will not affect other workspaces.  **In this tutorial, we'll create a new workspace to ensure our changes don't effect other's workloads.**
# MAGIC
# MAGIC #### 2.1) Create your own Tecton Workspace
# MAGIC Workspaces are created using the Tecton CLI. Let's create your own personal workspace and apply your local feature repo changes to it. To get started, create a workspace by running tecton workspace create YOUR_NAME. Your newly created workspace will be empty at first - you'll need to apply a feature repository to populate it with features.
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
# MAGIC **Tip:** For a complete list of workspace commands, simply run `tecton workspace -h`
# MAGIC
# MAGIC You can view and select different workspaces from the CLI with `tecton workspace list` and `tecton workspace select WORKSPACE_NAME`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2) Clone the Sample Feature Repo
# MAGIC In Tecton, [features are declared as code](https://docs.tecton.ai/examples/managing-feature-repos.html) in a **Tecton Feature Repository** which defines all of the features that Tecton will manage.  In a deployed live environment, the repo is integrated into CI/CD pipelines where updated and new features will go through a code review process.  In this tutorial we'll be directly registering your own features to your newly created workspace from the code repo hosted on your local machine.
# MAGIC
# MAGIC Before we build add a new feature, we'll need to clone the code repository used to build the existing features.  The sample feature repository for this demo can be found [here](https://github.com/tecton-ai-ext/tecton-sample-repo) -- if you already checked out this git repository to get a copy of this repo, you should already have the important files downloaded.  If not, clone the sample repository.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3) Add your feature to the repository
# MAGIC Add a new python file to the repo to define our new feature: `fraud/features/batch_features/cc_issuer.py` and place the following contents in it, representing the feature we created above.  
# MAGIC
# MAGIC ```python
# MAGIC from tecton import batch_feature_view
# MAGIC from fraud.entities import user
# MAGIC from fraud.data_sources.fraud_users import fraud_users_batch
# MAGIC from datetime import datetime, timedelta
# MAGIC
# MAGIC @batch_feature_view(
# MAGIC     sources=[fraud_users_batch],
# MAGIC     entities=[user],
# MAGIC     mode='spark_sql',
# MAGIC     online=True,
# MAGIC     offline=True,
# MAGIC     # Note the timestamp is the signup date, hence the old start_time.
# MAGIC     feature_start_time=datetime(2017, 1, 1),
# MAGIC     batch_schedule=timedelta(days=1),
# MAGIC     ttl=timedelta(days=3650),
# MAGIC     timestamp_field='signup_timestamp',
# MAGIC     tags={'release': 'production'},
# MAGIC     owner='mike.taveirne@tecton.ai',
# MAGIC     description='User credit card issuer derived from the user credit card number.',
# MAGIC )
# MAGIC def user_credit_card_issuer(fraud_users_batch):
# MAGIC     return f'''
# MAGIC         SELECT
# MAGIC             user_id,
# MAGIC             signup_timestamp,
# MAGIC             CASE SUBSTRING(CAST(cc_num AS STRING), 0, 1)
# MAGIC                 WHEN '4' THEN 'Visa'
# MAGIC                 WHEN '5' THEN 'MasterCard'
# MAGIC                 WHEN '6' THEN 'Discover'
# MAGIC                 ELSE 'other'
# MAGIC             END as credit_card_issuer
# MAGIC         FROM
# MAGIC             {fraud_users_batch}     
# MAGIC         '''
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4) Run `tecton apply`
# MAGIC Now that the code repo has been updated, it can be pushed out to the Tecton server instance.  Running a `tecton workspace list` should list workspaces on the server and show that the client is currently connected to the MY_WORKSPACE that was created above.  If not, issue a `tecton workspace select MY_WORKSPACE` to connect to it.  Now a `tecton plan` can be run to compare the code repo to the workspace, or a `tecton apply` to run the plan and prompt to push out changes.  Run `tecton apply` and hit y at the prompt to push in the changes.  The new feature should be detected and created.
# MAGIC
# MAGIC <pre>
# MAGIC ❯ tecton apply
# MAGIC Using workspace "MY_WORKSPACE" on cluster https://demo-peach.tecton.ai
# MAGIC ✅ Imported 52 Python modules from the feature repository
# MAGIC ✅ Collecting local feature declarations
# MAGIC ✅ Performing server-side feature validation: Finished generating plan.
# MAGIC  ↓↓↓↓↓↓↓↓↓↓↓↓ Plan Start ↓↓↓↓↓↓↓↓↓↓
# MAGIC
# MAGIC   + Create Transformation
# MAGIC     name:           user_credit_card_issuer
# MAGIC     owner:          mike.taveirne@tecton.ai
# MAGIC     description:    User credit card issuer derived from the user credit card number.
# MAGIC
# MAGIC   + Create Batch Feature View
# MAGIC     name:           user_credit_card_issuer
# MAGIC     owner:          mike.taveirne@tecton.ai
# MAGIC     description:    User credit card issuer derived from the user credit card number.
# MAGIC     warning:        This Feature View has online materialization enabled, but does not have monitoring configured. Use the `monitor_freshness` and `alert_email` fields to be alerted if there are issues with the materialization jobs.
# MAGIC     materialization: 10 backfills, 1 recurring batch job
# MAGIC     > backfill:     10 Backfill jobs 2016-12-31 00:00:00 UTC to 2023-04-07 00:00:00 UTC writing to both the Online and Offline Store
# MAGIC     > incremental:  1 Recurring Batch job scheduled every 1 day writing to both the Online and Offline Store
# MAGIC
# MAGIC  ↑↑↑↑↑↑↑↑↑↑↑↑ Plan End ↑↑↑↑↑↑↑↑↑↑↑↑
# MAGIC  Generated plan ID is 365a8930b9c64b7eb2d44afc0254588b
# MAGIC  View your plan in the Web UI: https://demo-peach.tecton.ai/app/prod/plan-summary/365a8930b9c64b7eb2d44afc0254588b
# MAGIC  ⚠️  Objects in plan contain warnings.
# MAGIC
# MAGIC Note: Updates to Feature Services may take up to 60 seconds to be propagated to the real-time feature-serving endpoint.
# MAGIC Are you sure you want to apply this plan? [y/N]> y
# MAGIC 🎉 all done!
# MAGIC </pre>
# MAGIC
# MAGIC Once the apply is done, the feature has now been registered in Tecton and can be discovered by other users on the platform.  The feature can also be used for consumption, and in a live workspace, features can be materialized to offline and online stores.  The example above was run against a live workspace to show the backfill materialization jobs that would be scheduled to load history of a feature and materialize it going forward as well.