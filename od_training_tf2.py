import os, sys
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  system("git clone --depth 1 https://github.com/tensorflow/models")


# Commented out IPython magic to ensure Python compatibility.
import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder

DATASET_PATH = 'person'

# UTILITIES
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  
  if not os.path.exists('output'):
      os.makedirs('output')
  if image_name:
    plt.imsave('output/' + image_name, image_np_with_annotations)
  # else:
  #   plt.imshow(image_np_with_annotations)

# DATA PREPARATION
# Load images and visualize
train_image_dir = 'models/research/object_detection/test_images/{}/train/'.format(DATASET_PATH)
train_images_np = []
for i in range(1, 10):
  image_path = os.path.join(train_image_dir, DATASET_PATH + str(i) + '.jpg')
  train_images_np.append(load_image_into_numpy_array(image_path))

plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['figure.figsize'] = [14, 7]

for idx, train_image_np in enumerate(train_images_np):
  plt.subplot(3, 7, idx+1)
  plt.imshow(train_image_np)
plt.show()

# ANNOTATE IMAGES WITH BOUNDING BOXES
gt_boxes = []
# If want to annotate manually
# colab_utils.annotate(train_images_np, box_storage_pointer=gt_boxes)
# If don't want to annotate
# gt_boxes = [np.asarray([[0.03938889, 0.145     , 1.        , 1.        ]]), np.asarray([[0.22438889, 0.21875   , 0.76272222, 0.82375   ]]), np.asarray([[0.17105555, 0.        , 0.94772222, 0.77      ]]), np.asarray([[0.15938889, 0.06916667, 0.87938889, 0.9425    ]]), np.asarray([[0.05438889, 0.03958333, 0.91772222, 0.87708333]]), np.asarray([[0.01438889, 0.00328947, 0.85272222, 0.56030702],
#        [0.46772222, 0.44736842, 1.        , 0.95614035]]), np.asarray([[0.12938889, 0.0025    , 0.99772222, 1.        ]]), np.asarray([[0.02772222, 0.24921875, 1.        , 0.86359375]]), np.asarray([[0.11772222, 0.08125   , 0.91772222, 0.92916667]]), np.asarray([[0.09772222, 0.18984375, 1.        , 0.62109375]]), np.asarray([[0.02772222, 0.0328125 , 1.        , 0.95109375]]), np.asarray([[0.13105555, 0.47296875, 0.83438889, 0.86671875]]), np.asarray([[0.19438889, 0.05833333, 0.82438889, 0.93333333]]), np.asarray([[0.35605555, 0.05083333, 0.81772222, 0.95866666]]), np.asarray([[0.17605555, 0.2784375 , 1.        , 0.72921875]]), np.asarray([[0.10772222, 0.00416667, 0.90438889, 0.47291667],
#        [0.11105555, 0.50416667, 0.89438889, 0.99583333]]), np.asarray([[0.07605555, 0.25625   , 1.        , 0.70625   ]]), np.asarray([[0.00438889, 0.18958333, 1.        , 0.93125   ]]), np.asarray([[0.10605555, 0.034375  , 1.        , 0.78515625]]), np.asarray([[0.06772222, 0.38916667, 0.94772222, 0.77666667]])]

gt_boxes = [np.asarray([[0.19605555, 0.0734375 , 0.68605555, 0.2640625 ],
       [0.12605555, 0.5765625 , 0.74938889, 0.74375   ],
       [0.33272222, 0.8890625 , 0.49772222, 0.94375   ],
       [0.35105555, 0.7890625 , 0.49105555, 0.8265625 ],
       [0.36605555, 0.7328125 , 0.47772222, 0.7703125 ]]), np.asarray([[0.31938889, 0.4125    , 0.81105555, 0.5640625 ],
       [0.21772222, 0.684375  , 0.78605555, 0.8390625 ]]), np.asarray([[0.30605555, 0.5703125 , 0.79272222, 0.7359375 ]]), np.asarray([[0.35438889, 0.625     , 0.93105555, 0.815625  ],
       [0.10938889, 0.3890625 , 0.90105555, 0.6359375 ]]), np.asarray([[0.34772222, 0.296875  , 0.76272222, 0.415625  ],
       [0.37272222, 0.2796875 , 0.64605555, 0.3359375 ],
       [0.14772222, 0.7671875 , 0.85272222, 0.8765625 ],
       [0.03105555, 0.5890625 , 0.94605555, 0.8203125 ]]), np.asarray([[0.29438889, 0.7875    , 0.99272222, 0.9765625 ],
       [0.31438889, 0.675     , 0.98438889, 0.809375  ],
       [0.43772222, 0.503125  , 0.97438889, 0.6453125 ],
       [0.32105555, 0.384375  , 0.98105555, 0.540625  ],
       [0.35438889, 0.2671875 , 0.98272222, 0.4140625 ],
       [0.27105555, 0.096875  , 0.97438889, 0.2703125 ],
       [0.21105555, 0.0015625 , 0.60272222, 0.0828125 ]]), np.asarray([[0.21438889, 0.0140625 , 0.66772222, 0.13125   ],
       [0.23105555, 0.2484375 , 0.67772222, 0.3578125 ],
       [0.21772222, 0.3609375 , 0.68105555, 0.4921875 ],
       [0.28438889, 0.46875   , 0.69605555, 0.603125  ],
       [0.33272222, 0.6953125 , 0.67272222, 0.7890625 ],
       [0.21938889, 0.5640625 , 0.68272222, 0.68125   ]]), np.asarray([[0.06105555, 0.190625  , 0.79605555, 0.3734375 ],
       [0.15105555, 0.4078125 , 0.71105555, 0.5359375 ],
       [0.11772222, 0.6203125 , 0.71605555, 0.75      ],
       [0.16772222, 0.7953125 , 0.54272222, 0.9015625 ],
       [0.20105555, 0.9078125 , 0.49438889, 0.9734375 ],
       [0.26438889, 0.578125  , 0.40272222, 0.6140625 ]]), np.asarray([[0.22105555, 0.0828125 , 0.90438889, 0.2734375 ],
       [0.19938889, 0.3328125 , 0.88272222, 0.484375  ],
       [0.20438889, 0.5296875 , 0.87438889, 0.7359375 ],
       [0.36438889, 0.7796875 , 0.85938889, 0.9015625 ]])]
print(len(gt_boxes))
# sys.exit(0)

# PREPARE DATA FOR TRAINING
# By convention, our non-background classes start counting at 1.
class1_id = 1
num_classes = 1

category_index = {class1_id: {'id': class1_id, 'name': DATASET_PATH}}

# CONVERT CLASS LABELS TO ONE-HOT; CONVERT EVERYTHING TO TENSORS
label_id_offset = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
  train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))
  gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
  zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
  gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))
print('Done prepping data.')

# # Let's just visualize the dog images as a sanity check
# # dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%
# dummy_scores = None
# plt.figure(figsize=(30, 15))
# for idx in range(20):
#   plt.subplot(3, 7, idx+1)
#   plot_detections(
#       train_images_np[idx],
#       gt_boxes[idx],
#       np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32),
#       dummy_scores, 
#       category_index)
# plt.show()

# # Create model and restore weights for all but last layer
# # Download the checkpoint and put it into models/research/object_detection/test_data/
# os.system("wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")
# os.system("tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")
# os.system("mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/")

tf.keras.backend.clear_session()
print('Building model and restoring weights for fine-tuning...', flush=True)
num_classes = 1
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'

# LOAD PIPELINE CONFIG AND BUILD A DETECTION MODEL.
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=model_config, is_training=True)

# Set up object-based checkpoint restore --- RetinaNet has two prediction
# `heads` --- one for classification, the other for box regression.  
# We will restore the box regression head but initialize the classification head
# from scratch (we show the omission below by commenting out the line that
# we would add if we wanted to restore both heads)
fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(checkpoint_path).expect_partial()

# Run model through a dummy image so that variables are created
image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')

# EAGER MODE CUSTOM TRAINING LOOP
tf.keras.backend.set_learning_phase(True)

batch_size = 8
learning_rate = 0.01
num_batches = 50

# Select variables in top layers to fine-tune.
trainable_variables = detection_model.trainable_variables
# print(len(trainable_variables))

to_fine_tune = []
prefixes_to_train = [
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

for var in trainable_variables:
  if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
    to_fine_tune.append(var)

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      preprocessed_images = tf.concat(
          [detection_model.preprocess(image_tensor)[0]
           for image_tensor in image_tensors], axis=0)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

  return train_step_fn

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
train_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune)

print('Start fine-tuning!', flush=True)
for idx in range(num_batches):
  # Grab keys for a random subset of examples
  all_keys = list(range(len(train_images_np)))
  random.shuffle(all_keys)
  example_keys = all_keys[:batch_size]

  # Note that we do not do data augmentation in this demo.  If you want a
  # a fun exercise, we recommend experimenting with random horizontal flipping
  # and random cropping :)
  gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
  gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
  image_tensors = [train_image_tensors[key] for key in example_keys]

  # Training step (forward pass + backwards pass)
  total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

  if idx % 10 == 0:
    print('batch ' + str(idx) + ' of ' + str(num_batches)
    + ', loss=' +  str(total_loss.numpy()), flush=True)

print('Done fine-tuning!')

# LOAD TEST IMAGES AND RUN INFERENCE WITH NEW MODEL!
test_image_dir = 'models/research/object_detection/test_images/{}/test/'.format(DATASET_PATH)
test_images_np = []
for i in range(1, 13):
  image_path = os.path.join(test_image_dir, 'test' + str(i) + '.jpg')
  test_images_np.append(np.expand_dims(
      load_image_into_numpy_array(image_path), axis=0))

# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor):
  """Run detection on an input image.

  Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  return detection_model.postprocess(prediction_dict, shapes)

# Note that the first frame will trigger tracing of the tf.function, which will
# take some time, after which inference should be fast.

label_id_offset = 1
for i in range(len(test_images_np)):
  input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
  detections = detect(input_tensor)

  plot_detections(
      test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32)+label_id_offset,
      detections['detection_scores'][0].numpy(),
      category_index, 
      figsize=(15, 20), 
      image_name="gif_frame_" + ('%02d' % i) + ".jpg")

# MAKE GIF FILE OF OUTPUT IMAGES
# imageio.plugins.freeimage.download()
# anim_file = 'dog_test.gif'
# filenames = glob.glob('gif_frame_*.jpg')
# filenames = sorted(filenames)
# last = -1
# images = []
# for filename in filenames:
#   image = imageio.imread(filename)
#   images.append(image)
# imageio.mimsave(anim_file, images, 'GIF-FI', fps=2)
# display(IPyImage(open(anim_file, 'rb').read()))