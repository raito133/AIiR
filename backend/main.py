import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from engine import ClassificationEngine
import pandas

# na razie uploadowane rzeczy lądują w /files
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'files'))
# można przefiltrować pliki przed uploadem
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# mozna wrzucić bezpośrednio przez '/', lub normalnie POSTem
@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Classification engine</title>
    <h1>Upload file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/spark", methods = ['GET', 'POST'])
def spark_task():
    if request.method == 'POST':
        testError, predictions = classification_engine.trainModel('files/' + request.form['text'])
        
        return render_template('results.html', tables = [predictions.to_html(classes='data')], titles=predictions.columns.values)
        

    return '''
    <!doctype html>
    <title>Classification engine</title>
    <h1>Start task</h1>
    <form method=post enctype=multipart/form-data>
      <input type=text name=text>
      <input type=submit value=Submit>
    </form>
    '''


if __name__ == "__main__":
    global classification_engine
    classification_engine = ClassificationEngine()    
    
    app.run(debug=True)
