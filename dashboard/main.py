import logging

import flask
import flask_cors

from routes.annotator import app_annotate
from routes.preprocess import app_preprocess
from routes.train import app_train

app = flask.Flask(__name__)
app.register_blueprint(app_preprocess)
app.register_blueprint(app_train)
app.register_blueprint(app_annotate)
flask_cors.CORS(app, resources={r"/*": {"origins": "*"}})


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


@app.route('/')
def index():
    return flask.render_template('index.html')


def validator():
    # Validating data (visually)
    raise NotImplementedError()


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    configure_logger()
    app.run(host='127.0.0.1', port=8080, debug=True)
