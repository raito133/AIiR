from flask import Flask, url_for
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
import os
from celery import Celery

# na razie uploadowane rzeczy lądują w /files
from werkzeug.utils import redirect

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'files'))

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()



def create_app():
    app = Flask(__name__)

    # na razie uploadowane rzeczy lądują w /files
    app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['TESTING'] = False
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    

    db.init_app(app)

    login_manager = LoginManager(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'


    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized_handler():
        return redirect(url_for('auth.login'))

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
