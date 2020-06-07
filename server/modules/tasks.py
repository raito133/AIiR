from celery import Celery
import os
celery = Celery(__name__, backend='redis://localhost:6379/0', broker='redis://localhost:6379/0')

@celery.task(bind=True)
def spark_job_task(self):
    task_id = self.request.id

    master_path = 'local[2]' # spark://192.168.1.30:7077

    project_dir = ''

    jar_path = '~/spark/jars/elasticsearch-hadoop-2.1.0.jar'

    spark_code_path = project_dir + 'classification.py'

    os.system("~/spark/bin/spark-submit --master %s --jars %s %s %s" %
              (master_path, jar_path, spark_code_path, task_id))

    return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': 42}
 
