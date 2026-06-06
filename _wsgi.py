"""WSGI entry used by `label-studio-ml start .` and `python _wsgi.py`.

You normally don't need to edit this file.
"""
import os

from label_studio_ml.api import init_app

from model import AnnotationBackend

app = init_app(model_class=AnnotationBackend)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "9090")), debug=False)
