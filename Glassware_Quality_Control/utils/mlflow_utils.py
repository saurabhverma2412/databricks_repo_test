# Databricks notebook source
import mlflow
import mlflow.mleap
import mlflow.spark
mlflowclient = mlflow.tracking.MlflowClient()

# COMMAND ----------

def best_run(mlflow_experiment_id, mlflow_search_query):
  mlflowclient = mlflow.tracking.MlflowClient()
  best_run = None
  runs = mlflowclient.search_runs([mlflow_experiment_id], mlflow_search_query)
  for run in runs:
    if best_run is None or run.data.metrics[model_compare_metric] > best_run[1]:
      best_run = (run.info.run_uuid,run.data.metrics[model_compare_metric])
  best_runid = best_run[0]
  
  best_run_details = {}
  best_run_details['runid'] = best_runid
  best_run_details['params'] = mlflowclient.get_run(best_runid).to_dictionary()["data"]["params"]
  best_run_details['metrics'] = mlflowclient.get_run(best_runid).to_dictionary()["data"]["metrics"]
  
  artifact_uri = mlflowclient.get_run(best_runid).to_dictionary()["info"]["artifact_uri"]
#   best_run_details['confusion_matrix_uri'] = "/" + artifact_uri.replace(":","") + "/confusion_matrix.pkl"
  best_run_details['confusion_matrix_uri'] = "runs:/" + best_runid+ "/artifacts/confusion_matrix.pkl"
#   best_run_details['spark-model'] = "/" + artifact_uri.replace(":","") + "/spark-model"
  best_run_details['spark-model'] = "runs:/" + best_runid+ "/spark-model"
  
  return best_run_details

# COMMAND ----------

def run_exists(mlflow_experiment_id, params):
  mlflow_search_query = ' and '.join([f'params.{key} = \'{value}\'' for key, value in params.items()])
  runs = mlflowclient.search_runs([mlflow_experiment_id], mlflow_search_query)
  if len(runs) > 0: return True
  return False

# COMMAND ----------

def get_run_details(runid):
  run_details = {}
  run_details['runid'] = runid
  run_details['params'] = mlflowclient.get_run(runid).to_dictionary()["data"]["params"]
  run_details['metrics'] = mlflowclient.get_run(runid).to_dictionary()["data"]["metrics"]
  
  artifact_uri = mlflowclient.get_run(runid).to_dictionary()["info"]["artifact_uri"]
#   run_details['confusion_matrix_uri'] = "/" + artifact_uri.replace(":","") + "/confusion_matrix.pkl"
  run_details['confusion_matrix_uri'] = "runs:/" + runid+ "/artifacts/confusion_matrix.pkl"
#   run_details['spark-model'] = "/" + artifact_uri.replace(":","") + "/spark-model"
  run_details['spark-model'] = "runs:/" + runid+ "/spark-model"
  
  return run_details

# COMMAND ----------

def get_model_production(mlflow_experiment_id):
  mlflow_search_query = "tags.state='production'"
  run = mlflowclient.search_runs([mlflow_experiment_id], mlflow_search_query)
  if run:
    runid = run[0].info.run_uuid
    return get_run_details(runid)
  else:
    return 0

# COMMAND ----------

def push_model_production(mlflow_experiment_id, runid, userid, start_date):
  
  if runid!=0:
    prod_run_details = get_model_production(mlflow_experiment_id)
    if prod_run_details!=0:
      terminate_model_production(prod_run_details['runid'], userid, start_date)

    mlflowclient.set_tag(runid, 'state', 'production')
    mlflowclient.set_tag(runid, 'production_marked_by', userid)
    mlflowclient.set_tag(runid, 'production_start', start_date)
    mlflowclient.set_tag(runid, 'production_end', '')
    return True
  else:
    return False

# COMMAND ----------

def terminate_model_production(runid, userid, end_date):
  if runid!=0:
    mlflowclient.set_tag(runid, 'state', 'ex_production')
    mlflowclient.set_tag(runid, 'production_marked_by', userid)
    mlflowclient.set_tag(runid, 'production_end', end_date)
    return True
  else:
    return False
