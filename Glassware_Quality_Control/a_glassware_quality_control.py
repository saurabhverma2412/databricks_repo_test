# Databricks notebook source
# MAGIC %md 
# MAGIC # Glassware Quality Control

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective: 
# MAGIC - To predict and grade the quality of the product as they are made, prior to actual inspection process which are a day or two later.
# MAGIC - This could act as an early indicator of quality for process optimization, and to send lower quality products for additional inspection and categorization.
# MAGIC 
# MAGIC ### About Data:
# MAGIC - sensor_reading - Sensor data from manufacturing equipment is streamed through IoT devices. Final aggregated data includes Temperature, and Pressure, along with process duration.
# MAGIC - product_quality - For each product id, actual quality categorization is available

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Architecture Reference
# MAGIC 
# MAGIC <!-- <img src="https://publicimg.blob.core.windows.net/images/model_drift_architecture.png" width="1300"> -->
# MAGIC <img src="https://joelcthomas.github.io/modeldrift/img/model_drift_architecture.png" width="1300">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### End to End Pipeline  
# MAGIC ### Data Access -> Data Prep -> Model Training -> Deployment -> Monitoring -> Action & Feedback Loop

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md ## <a>Data Access & Prep</a>

# COMMAND ----------

# MAGIC %md
# MAGIC __Setup access to blob storage, where data is streamed from and to, in delta lake__

# COMMAND ----------

# MAGIC %run ./data/data_access

# COMMAND ----------

# MAGIC %md
# MAGIC __Read sensor_reading & product_quality, join and prepare the data to be fed for model training__

# COMMAND ----------

# MAGIC %run ./data/data_prep

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assume today is 2019-07-11  
# MAGIC __Using data from (2019-07-01 - 2019-07-10) to generate models, for which sensor_reading and product_quality data are available__  
# MAGIC   
# MAGIC __To reproduce this demo, generate and push initial set of data using ./demo_utils/demo_data_init __

# COMMAND ----------

# MAGIC %run ./demo_utils/demo_data_init

# COMMAND ----------

# MAGIC %md
# MAGIC __Prior to production workflow, typically one could setup a notebook for EDA, get familiar with dataset, and explore modeling methods__  
# MAGIC __Check EDA notebook example here, for this project__

# COMMAND ----------

today_date = '2019-07-11 18:00'

# COMMAND ----------

model_data_date = {'start_date':'2019-07-01 00:00:00', 'end_date':'2019-07-10 23:59:00'}
model_df = model_data(model_data_date)
display(model_df)

# COMMAND ----------

# MAGIC %md ## <a>Model Training & Tuning</a>

# COMMAND ----------

# MAGIC %md
# MAGIC __Run various models (Random Forest, Decision Tree, XGBoost), each with its own set of hyperparameters, and log all the information to MLflow__

# COMMAND ----------

# MAGIC %run ./glassware_quality_modeler/generate_models

# COMMAND ----------

best_rf_run

# COMMAND ----------

# DBTITLE 1,Fix for the confusion matrix pickle file
# Download and then upload the confusion matrix pickle file an put it to the underlying system path because pandas can't access databricks path directly.
# dbutils.fs.ls('/FileStore/tables/confusion_matrix.pkl')
# dbutils.fs.cp('/FileStore/tables/confusion_matrix.pkl', 'file:/dbfs/mnt/artifacts/confusion_matrix.pkl', True)
# dbutils.fs.ls('file:/dbfs/mnt/artifacts')

# COMMAND ----------

# MAGIC %md ## <a>Model Selection & Deployment</a>

# COMMAND ----------

# MAGIC %md
# MAGIC __Search MLflow to find the best model from above experiment runs across all model types__

# COMMAND ----------

mlflow_search_query = "params.model_data_date = '"+ model_data_date['start_date']+ ' - ' + model_data_date['end_date']+"'"
best_run_details = best_run(mlflow_exp_id, mlflow_search_query)

print("Best run from all trials:" + best_run_details['runid'])
print("Params:")
print(best_run_details["params"])
print("Metrics:")
print(best_run_details["metrics"])

# COMMAND ----------

# MAGIC %md
# MAGIC __Mark the best run as production in MLflow, to be used during scoring__

# COMMAND ----------

push_model_production(mlflow_exp_id, best_run_details['runid'], userid, today_date)

# COMMAND ----------

# MAGIC %md ## <a>Model Scoring</a>

# COMMAND ----------

get_model_production(mlflow_exp_id)

# COMMAND ----------

# MAGIC %run ./glassware_quality_scorer/score_quality

# COMMAND ----------

# MAGIC %md
# MAGIC __Read the sensor_reading stream, apply model scoring, and write the output stream as 'predicted_quality' delta table__

# COMMAND ----------

sensor_reading_stream = stream_sensor_reading()
predict_stream = stream_score_quality(sensor_reading_stream)

# COMMAND ----------

# def get_run_details(runid):
#   run_details = {}
#   run_details['runid'] = runid
#   run_details['params'] = mlflowclient.get_run(runid).to_dictionary()["data"]["params"]
#   run_details['metrics'] = mlflowclient.get_run(runid).to_dictionary()["data"]["metrics"]
  
#   artifact_uri = mlflowclient.get_run(runid).to_dictionary()["info"]["artifact_uri"]
#   run_details['confusion_matrix_uri'] = "/" + artifact_uri.replace(":","") + "/confusion_matrix.pkl"
# #   run_details['spark-model'] = "/" + artifact_uri.replace(":","") + "/spark-model"
#   run_details['spark-model'] = "runs:/" + runid+ "/spark-model"
  
#   return run_details

# model_runid_test = '99e56f0fea104f62a393fcf320b2cf84'
# run_details = get_run_details(model_runid_test)
# print('Using model version:'+ run_details['runid'])
# print('spark-model path:' + run_details['spark-model'])
# prod_model = mlflow.spark.load_model(run_details['spark-model'])

# COMMAND ----------

# MAGIC %md ## <a>Model Monitoring & Feedback</a>

# COMMAND ----------

# MAGIC %run ./model_quality/model_quality_monitor

# COMMAND ----------

predicted_quality = get_predicted_quality()
product_quality = get_product_quality()
model_quality_summary = track_model_quality(product_quality, predicted_quality)

# COMMAND ----------

display(model_quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assume today is 2019-07-16

# COMMAND ----------

plot_model_quality(model_quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assume today is 2019-07-21
# MAGIC __To reproduce this demo, push data for range 2019-07-16 - 2019-07-21 using ./demo_utils/demo_data_init __

# COMMAND ----------

today_date = '2019-07-21 01:00'

# COMMAND ----------

#%run ./demo_utils/demo_data_init

# COMMAND ----------

model_quality_summary = track_model_quality(get_product_quality(), get_predicted_quality())
plot_model_quality(model_quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC We see drift occurs on 07/20 (based on process date)

# COMMAND ----------

# MAGIC %md ## <a>Retrain After Drift</a>

# COMMAND ----------

# MAGIC %md
# MAGIC __Read sensor_reading & product_quality, join and prepare the data to be fed for model training__

# COMMAND ----------

model_data_date = {'start_date':'2019-07-19 00:00:00', 'end_date':'2019-07-21 00:00:00'}
model_df = model_data(model_data_date)
display(model_df)

# COMMAND ----------

# MAGIC %md
# MAGIC __Run various models (Random Forest, Decision Tree, XGBoost), each with its own set of hyperparameters, and log all the information to MLflow__

# COMMAND ----------

# MAGIC %run ./glassware_quality_modeler/generate_models

# COMMAND ----------

# MAGIC %md
# MAGIC __Search MLflow to find the best model from above experiment runs across all model types__

# COMMAND ----------

mlflow_search_query = "params.model_data_date = '"+ model_data_date['start_date']+ ' - ' + model_data_date['end_date']+"'"
best_run_details = best_run(mlflow_exp_id, mlflow_search_query)

print("Best run from all trials:" + best_run_details['runid'])
print("Params:")
print(best_run_details["params"])
print("Metrics:")
print(best_run_details["metrics"])

# COMMAND ----------

# MAGIC %md
# MAGIC __Mark the best run as production in MLflow, to be used during scoring__

# COMMAND ----------

push_model_production(mlflow_exp_id, best_run_details['runid'], userid, today_date)

# COMMAND ----------

predict_stream.stop()
predict_stream = stream_score_quality(stream_sensor_reading())

# COMMAND ----------

# MAGIC %md
# MAGIC __Summary after retrain__

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assume today is 2019-08-01
# MAGIC __To reproduce this demo, push data for range 2019-07-21 - 2019-08-01 using ./demo_utils/demo_data_init __

# COMMAND ----------

model_quality_summary = track_model_quality(get_product_quality(), get_predicted_quality())
plot_model_quality(model_quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assume today is 2019-08-12
# MAGIC __To reproduce this demo, push data for range > 2019-08-01 using ./demo_utils/demo_data_init __

# COMMAND ----------

model_quality_summary = track_model_quality(get_product_quality(), get_predicted_quality())
plot_model_quality(model_quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC __ Let's see what would have happened if model was not updated?__

# COMMAND ----------

predicted_quality = score_quality(get_sensor_reading(), '396adf449c6146a8b37ac7e679c2e3d1')
model_quality_summary = track_model_quality(get_product_quality(), predicted_quality)
plot_model_quality(model_quality_summary)

# COMMAND ----------

# MAGIC %md ## <a>Summary</a>

# COMMAND ----------

predicted_quality = score_quality(get_sensor_reading(), '99e56f0fea104f62a393fcf320b2cf84')
model_quality_summary_1 = track_model_quality(get_product_quality(), predicted_quality)
predicted_quality = score_quality(get_sensor_reading(), '396adf449c6146a8b37ac7e679c2e3d1')
model_quality_summary_2 = track_model_quality(get_product_quality(), predicted_quality)

# COMMAND ----------

plot_summary(model_quality_summary_1, model_quality_summary_2)
