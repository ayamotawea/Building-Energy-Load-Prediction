import dagshub
dagshub.init(repo_owner='ayamotawea', repo_name='Building-Energy-Load-Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)