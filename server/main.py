import os
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, Blueprint, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename, redirect
import pandas as pd
from celery import Celery
from elasticsearch import Elasticsearch
from sqlalchemy import update
from datetime import datetime

from server import db
from server.models import User, Task
from server.modules import tasks as t

main = Blueprint('main', __name__)

# na razie uploadowane rzeczy lądują w /files
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'files'))
# można przefiltrować pliki przed uploadem
ALLOWED_EXTENSIONS = {'csv'}

ES_HOST = {
    "host": "localhost",
    "port": 9200
}
es = Elasticsearch(hosts=[ES_HOST])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# mozna wrzucić bezpośrednio przez '/', lub normalnie POSTem
@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    print('sesja: ' + str(session['email']))
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
    # if task_id in user.tasks
    #     do
    # else
    #     don't
    user = User.query.filter_by(email=session['email']).first()
    if str(task_id) not in user.gettasks():
        return render_template('profile.html')

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

    task_id_str = str(task.id)
    user = User.query.filter_by(email=session['email']).first()
    user.tasks += user.tasks + task_id_str + ";"
    db.session.commit()

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
        'id_string': task_id_str,
        'status': 'Spark job pending..',
        'start_time': datetime.utcnow()
    })
    link = 'http://localhost:5000/spark_task/' + task_id_str + 'output.csv'
    new_task = Task(user_id=user.id, id_string = link)
    db.session.add(new_task)
    db.session.commit()
    return jsonify({}), 202, {'Location': url_for('main.taskstatus', task_id=task.id)}


@main.route("/spark_task/<result>", methods=['GET'])
@login_required
def result(result):
    print('>>>>>>session: ' + str(session['email']))
    user = User.query.filter_by(email=session['email']).first()
    if str(result[:36]) not in user.gettasks():
        return render_template('profile.html')

    predictions = pd.read_csv(result)
    return render_template('results.html', tables=[predictions.to_html(classes='data')],
                           titles=predictions.columns.values)

@main.route("/history", methods=['GET'])
@login_required
def history():
    tasks = []
    user = User.query.filter_by(email=session['email']).first()
    tasks_db = Task.query.filter_by(user_id=user.id)
    for t in tasks_db:
        tasks.append(t.id_string)
    return render_template('history.html', tasks=tasks)