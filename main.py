from flask import Blueprint, render_template, request, flash, url_for
from flask_login import login_required, current_user
from werkzeug.utils import redirect, secure_filename
import pandas

from . import engine
from . import db
import os

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'files'))

# można przefiltrować pliki przed uploadem
ALLOWED_EXTENSIONS = {'csv'}
classification_engine = engine.ClassificationEngine()

main = Blueprint('main', __name__)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/')
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


@main.route("/spark", methods=['GET', 'POST'])
@login_required
def spark_task():
    if request.method == 'POST':
        testError, predictions = classification_engine.trainModel('swiotp/files/' + request.form['text'])

        return render_template('results.html', tables=[predictions.to_html(classes='data')],
                               titles=predictions.columns.values)

    return '''
    <!doctype html>
    <title>Classification engine</title>
    <h1>Start task</h1>
    <form method=post enctype=multipart/form-data>
      <input type=text name=text>
      <input type=submit value=Submit>
    </form>
    '''
