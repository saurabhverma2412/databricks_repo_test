# Databricks notebook source
# DBTITLE 1,Run Random Forest
# MAGIC %run ./model_random_forest

# COMMAND ----------

print("Modeling using Random Forest:")
best_rf_run = run_randomforest(model_df)

# COMMAND ----------

print("Best run within Random Forest trials:" + best_rf_run['runid'])
print("Params:")
print(best_rf_run["params"])
print("Metrics:")
print(best_rf_run["metrics"])

# COMMAND ----------



# COMMAND ----------

# display(plot_confusion_matrix(best_rf_run['confusion_matrix_uri']))
# display(plot_confusion_matrix('file:/dbfs/mnt/artifacts/confusion_matrix_random_forest.pkl'))


# COMMAND ----------

# DBTITLE 1,Run Decision Tree
# MAGIC %run ./model_decision_tree

# COMMAND ----------

print("Modeling using Decision Tree:")
best_dt_run = run_decisiontree(model_df)

# COMMAND ----------

print("Best run within Random Forest trials:" + best_dt_run['runid'])
print("Params:")
print(best_dt_run["params"])
print("Metrics:")
print(best_dt_run["metrics"])

# COMMAND ----------

# display(plot_confusion_matrix(best_dt_run['confusion_matrix_uri']))
# display(plot_confusion_matrix('file:/dbfs/mnt/artifacts/confusion_matrix_decision_tree.pkl'))

# COMMAND ----------

#%run ./model_xgboost
