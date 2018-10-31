import skimage.io
import json
import tensorflow as tf
from flask import Flask, request
from util import get_model, transform


def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response


app = Flask(__name__)
app.after_request(after_request)
model = get_model()
global graph
graph = tf.get_default_graph()


@app.route('/')
def hello():
    return 'This is a empty page.'


@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    image = skimage.io.imread(image)
    with graph.as_default():
        results = model.detect([image], verbose=0)
    boxes = results[0]['rois']
    print(boxes[0])
    formatted_boxes = list(map(transform, boxes))
    return json.dumps(formatted_boxes)


if __name__ == '__main__':
    app.run(port=40022, host='0,0,0,0')
