import os

from detect_object import get_model
import skimage.io

model = get_model()
model = get_model()
model = get_model()
image = skimage.io.imread('./images/road.jpg')
results = model.detect([image], verbose=1)