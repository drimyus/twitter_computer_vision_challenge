import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from utils import visualization_utils as vis_util
from utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool

MAX_NUMBER_OF_BOXES = 20
MINIMUM_CONFIDENCE = 0.5

PATH_TO_LABELS = '../annotations/label_map.pbtxt'
PATH_TO_TEST = '../test'
PATH_TO_TEST_IMAGES_DIR = os.path.join(PATH_TO_TEST, 'test_images')
PATH_TO_TEST_OUTPUT = os.path.join(PATH_TO_TEST, 'output')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize,
                                                            use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = '../output_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_np_expanded})

    # import cv2
    # cv_img = cv2.imread(image_path)
    # h, w = cv_img.shape[:2]
    # for i, box in enumerate(np.reshape(boxes, (100, 4)).tolist()):
    #     y0, x0, y2, x2 = (np.array(box) * np.array([h, w, h, w])).astype(np.int)
    #     label = categories[int(classes[0][i]-1)]['name']
    #     score = scores[0][i]
    #
    #     if x0 < 0.5 * w:
    #         continue
    #
    #     print("label:", label, "position: [", x0, y0, x2, y2, "] score:", round(score * 100, 1), "%")
    #     cv2.rectangle(cv_img, (x0, y0), (x2, y2), (0, 0, 255))
    #     cv2.putText(cv_img, f"{label}_{round(score * 100)}%", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
    #                 cv2.LINE_AA)
    #
    # cv2.imshow("cv", cv2.resize(cv_img, None, fx=1.0, fy=1.0))
    # cv2.imwrite(os.path.join(PATH_TO_TEST_OUTPUT, os.path.basename(image_path)), cv_img)
    # cv2.waitKey(0)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
    fig = plt.figure()
    fig.set_size_inches(9, 12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image_np, aspect='auto')
    plt.savefig(os.path.join(PATH_TO_TEST_OUTPUT, os.path.basename(image_path)), dpi=62)
    plt.close(fig)


# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))


# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in TEST_IMAGE_PATHS:
            print(image_path)
            detect_objects(image_path)
