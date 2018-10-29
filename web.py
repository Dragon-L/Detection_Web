from flask import Flask, request
import skimage.io
from util import get_model

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/detect')
def detect():
    image_path = request.args.get('image_path', '', type=str)
    image = skimage.io.imread(image_path)
    model = get_model()
    results = model.detect([image], verbose=1)
    return results

if __name__ == '__main__':
    app.run(port=40022)
