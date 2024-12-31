# retrain.py - Optimized for TensorFlow 2.x Compatibility

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile
import logging

import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf  # Use TensorFlow 1.x compatibility mode
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors

from PIL import Image  # Added missing import

# Set up Python's logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = None

# Parameters tied to the model architecture
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system."""
    if not tf.io.gfile.exists(image_dir):
        logger.error("Image directory '%s' not found.", image_dir)
        return None

    result = collections.OrderedDict()
    sub_dirs = [os.path.join(image_dir, item) for item in tf.io.gfile.listdir(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs if tf.io.gfile.isdir(item))

    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        logger.info("Looking for images in '%s'", dir_name)
        for extension in extensions:
            file_glob = os.path.join(sub_dir, '*.' + extension)
            file_list.extend(tf.io.gfile.glob(file_glob))
        if not file_list:
            logger.warning('No files found in %s', sub_dir)
            continue
        if len(file_list) < 20:
            logger.warning('WARNING: Folder %s has less than 20 images, which may cause issues.', dir_name)
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            logger.warning(
                'WARNING: Folder %s has more than %d images. Some images will never be selected.',
                dir_name, MAX_NUM_IMAGES_PER_CLASS)

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # Ensure that each category has at least one image
        total_images = len(training_images) + len(testing_images) + len(validation_images)
        if total_images >= 3:
            if not validation_images:
                # Move one image from training to validation
                validation_images.append(training_images.pop())
                logger.info("Moved one image from training to validation for label '%s'.", label_name)
            if not testing_images:
                # Move one image from training to testing
                testing_images.append(training_images.pop())
                logger.info("Moved one image from training to testing for label '%s'.", label_name)
        elif total_images == 2:
            # Assign one image to training and one to validation
            if len(training_images) == 2:
                validation_images.append(training_images.pop())
            elif len(training_images) == 1 and len(testing_images) == 1:
                validation_images.append(testing_images.pop())
            elif len(testing_images) == 2:
                validation_images.append(testing_images.pop())
            logger.info("Assigned images to ensure at least one image in training and validation for label '%s'.", label_name)
        elif total_images == 1:
            # Assign the single image to training
            pass  # No action needed

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index."""
    if label_name not in image_lists:
        logger.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        logger.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        logger.fatal('Label %s has no images in the category %s.', label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture):
    """Returns a path to a bottleneck file for a label at the given index."""
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """Creates a graph from saved GraphDef file and returns a Graph object."""
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ])
    return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer."""
    # Decode and resize the image
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # Run through the recognition network
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract(data_url):
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                           (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        logger.info('Successfully downloaded %s (%d bytes).', filename, statinfo.st_size)
    # Extract the tar file
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    logger.info('Creating bottleneck at %s', bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not tf.io.gfile.exists(image_path):
        logger.fatal('File does not exist %s', image_path)
    image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """Retrieves or calculates bottleneck values for an image."""
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        logger.warning('Invalid float found in %s, recreating bottleneck.', bottleneck_path)
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """Ensures all the training, testing, and validation bottlenecks are cached."""
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    logger.info('%d bottleneck files created.', how_many_bottlenecks)


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
    """Retrieves bottleneck values for cached images."""
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        for _ in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor, architecture)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions."""
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)
        if not tf.io.gfile.exists(image_path):
            logger.fatal('File does not exist %s', image_path)
        jpeg_data = tf.io.gfile.GFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck_values)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags."""
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """Creates the operations to apply the specified distortions."""
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform([], minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform([], minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size):
    """Adds a new softmax and fully-connected layer for training."""
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    # Organize the ops as `final_training_ops` for easier visibility in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results."""
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
    """Saves the trained graph to a file."""
    output_node_names = [n.name for n in graph.as_graph_def().node]  # Collect all node names
    logger.info("Saving the following nodes in the graph: %s", output_node_names)

    # Convert variables to constants using TensorFlow's compatibility module
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(),
        [FLAGS.final_tensor_name],  # Replace with your output layer's name
    )

    # Save the frozen graph to a file
    with tf.io.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    logger.info('Saved graph to %s', graph_file_name)


def prepare_file_system():
    """Prepares necessary directories for training."""
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.makedirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it."""
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) not in [3, 4]:
            logger.error("Couldn't understand architecture name '%s'", architecture)
            return None
        version_string = parts[1]
        if version_string not in ['1.0', '0.75', '0.50', '0.25']:
            logger.error(
                "The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25', "
                "but found '%s' for architecture '%s'", version_string, architecture)
            return None
        size_string = parts[2]
        if size_string not in ['224', '192', '160', '128']:
            logger.error(
                "The Mobilenet input size should be '224', '192', '160', or '128', "
                "but found '%s' for architecture '%s'", size_string, architecture)
            return None
        is_quantized = False
        if len(parts) == 4:
            if parts[3] != 'quantized':
                logger.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_{}_{}_frozen.tgz'.format(version_string, size_string)
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        model_base_name = 'quantized_graph.pb' if is_quantized else 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_{}_{}'.format(version_string, size_string)
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        logger.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph."""
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def main(_):
    """Main function to run the retraining process."""
    # Set logging verbosity
    logger.setLevel(logging.INFO)

    # Prepare necessary directories
    prepare_file_system()

    # Gather information about the model architecture
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        logger.error('Did not recognize architecture flag')
        return -1

    # Set up the pre-trained graph
    maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info)

    # Create lists of all the images
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    if image_lists is None:
        logger.error('Image lists creation failed.')
        return -1

    class_count = len(image_lists.keys())
    if class_count == 0:
        logger.error('No valid folders of images found at %s', FLAGS.image_dir)
        return -1
    if class_count == 1:
        logger.error('Only one valid folder of images found at %s - multiple classes are needed for classification.', FLAGS.image_dir)
        return -1

    # Check if we need to apply any distortions
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        if do_distort_images:
            # Apply distortions
            distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
                FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                FLAGS.random_brightness, model_info['input_width'],
                model_info['input_height'], model_info['input_depth'],
                model_info['input_mean'], model_info['input_std'])
        else:
            # Cache bottleneck values
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, FLAGS.architecture)

        # Add the new training layer
        train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor = add_final_training_ops(
            len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor,
            model_info['bottleneck_tensor_size'])

        # Add evaluation step
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all summaries for TensorBoard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'train'), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'validation'))

        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        for i in range(FLAGS.how_many_training_steps):
            if do_distort_images:
                # Get distorted bottlenecks
                train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                # Get cached bottlenecks
                train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.architecture)

            # Run training step and write summaries
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Periodically evaluate training and validation accuracy
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                logger.info('%s: Step %d: Train accuracy = %.1f%%', datetime.now(), i, train_accuracy * 100)
                logger.info('%s: Step %d: Cross entropy = %f', datetime.now(), i, cross_entropy_value)

                validation_bottlenecks, validation_ground_truth, _ = get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.architecture)

                # Run validation step and write summaries
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                logger.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)',
                            datetime.now(), i, validation_accuracy * 100,
                            len(validation_bottlenecks))

            # Save intermediate graphs if needed
            intermediate_frequency = FLAGS.intermediate_store_frequency
            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                and i > 0):
                intermediate_file_name = os.path.join(
                    FLAGS.intermediate_output_graphs_dir,
                    'intermediate_' + str(i) + '.pb')
                logger.info('Save intermediate result to: %s', intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name)

        # Final test evaluation
        test_bottlenecks, test_ground_truth, test_filenames = get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture)
        test_accuracy, predictions = sess.run(
            [evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        logger.info('Final test accuracy = %.1f%% (N=%d)', test_accuracy * 100, len(test_bottlenecks))

        # Optionally print misclassified test images
        if FLAGS.print_misclassified_test_images:
            logger.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    logger.info('%70s  %s', test_filename, list(image_lists.keys())[predictions[i]])

        # Save the trained graph and labels
        save_graph_to_file(sess, graph, FLAGS.output_graph)
        with tf.io.gfile.GFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/Users/venom/Downloads/real_time_face_attendance/TrainingImage',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='/tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
           How many steps to store intermediate graph. If "0" then will not store.\
        """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=6000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
        Whether to print out a list of all misclassified test images.\
        """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
        The name of the output classification layer in the retrained graph.\
        """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
        Whether to randomly flip half of the training images horizontally.\
        """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
        A percentage determining how much of a margin to randomly crop off the
        training images.\
        """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly scale up the size of the
        training images by.\
        """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly multiply the training image
        input pixels up or down by.\
        """
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='inception_v3',
        help="""\
        Which model architecture to use. 'inception_v3' is the most accurate, but
        also the slowest. For faster or smaller models, choose a MobileNet with the
        form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
        'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
        pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
        less accurate, but smaller and faster network that's 920 KB on disk and
        takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
        for more information on MobileNet.\
        """)

    FLAGS, unparsed = parser.parse_known_args()

    main([sys.argv[0]] + unparsed)
```
# Ensure TensorFlow 2.x compatibility or update the script accordingly
