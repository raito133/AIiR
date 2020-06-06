import os
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, Blueprint
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename, redirect
import pandas as pd
from celery import Celery
from elasticsearch import Elasticsearch  

from datetime import datetime
from server.modules import tasks as t
main = Blueprint('main', __name__)

# na razie uploadowane rzeczy lądują w /files
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'files'))
# można przefiltrować pliki przed uploadem
ALLOWED_EXTENSIONS = {'csv'}



ES_HOST = {
    "host" : "localhost", 
    "port" : 9200
}
es = Elasticsearch(hosts = [ES_HOST])     


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# mozna wrzucić bezpośrednio przez '/', lub normalnie POSTem
@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Nie wybrano pliku')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.abspath(os.path.join(os.getcwd(), 'files/')) + filename)
            return render_template('profile.html')
    return '''
        <!doctype html>
        <title>Classification engine</title>
        <h1>Upload file</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''


@main.route('/dotask', methods=['GET'])
@login_required
def dotask():
    if request.method == 'GET':
        return render_template('task.html')


@main.route('/status/<task_id>')
@login_required
def taskstatus(task_id, methods=['GET']):
    task = t.spark_job_task.AsyncResult(task_id)

    if task.state == 'FAILURE':
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    else:
        # otherwise get the task info from ES
        es_task_info = es.get(index='spark-jobs', doc_type='job', id=task_id)
        response = es_task_info['_source']
        response['state'] = task.state

    return jsonify(response)

@main.route('/sparktask', methods=['POST'])
def sparktask():
    task = t.spark_job_task.apply_async()

    if not es.indices.exists('spark-jobs'):
        print("creating '%s' index..." % ('spark-jobs'))
        res = es.indices.create(index='spark-jobs', body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        })
        print(res)

    es.index(index='spark-jobs', doc_type='job', id=task.id, body={
        'current': 0,
        'total': 100,
        'status': 'Spark job pending..',
        'start_time': datetime.utcnow()
    })

    return jsonify({}), 202, {'Location': url_for('main.taskstatus', task_id=task.id)}


@main.route("/spark_task/<result>", methods=['GET'])
def result(result):
    predictions = pd.read_csv(result)
    return render_template('results.html', tables=[predictions.to_html(classes='data')],
                           titles=predictions.columns.values)
