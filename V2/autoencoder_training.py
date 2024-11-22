# imports
import argparse
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import configs
from utils.ops import load_image
from utils.ops import save_pkl_file

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a Convolutional ' \
                                                 ' autoencoder that\'ll be' \
                                                 'later used to vectorize' \
                                                 ' images.')
    parser.add_argument('--bottleneck_layer_size',
                        help='Size of bottleneck layer in autoencoder, play ' \
                             'with this parameter.',
                        nargs='?', const=6e3, default=6e3, type=int)
    parser.add_argument('--batch_size',
                        help='Batch size to use when training.',
                        nargs='?', const=16, default=16, type=int)
    parser.add_argument('--learning_rate',
                        help='The learning rate for AdamOptimizer.',
                        nargs='?', const=1e-3, default=1e-3, type=float)
    parser.add_argument('--end_step',
                        help='The step the training process will end at.',
                        nargs='?', const=1e3, default=1e3, type=float)
    parser.add_argument('--saving_interval',
                        help='After how many steps it will save a new ' \
                             'checkpoint of the model.',
                        nargs='?', const=1e2, default=1e2, type=float)
    parser.add_argument('--max_ckpts_to_keep',
                        help='The maximum amount of checkpoints to have in ' \
                             'training folder at once.',
                        nargs='?', const=5, default=5, type=int)
    parser.add_argument('--images_path',
                        help='The path to the images to train on.',
                        nargs='?', const='images', default='images')
    parser.add_argument('--training_path',
                        help='The path to the directory where checkpoints ' \
                             'and tensorboard summaries lie.',
                        nargs='?', const='training', default='training')
    parser.add_argument('--replace_img_tfrecord',
                        help='If the program should update the tfrecord file ' \
                             'with image data if there is data, yes or no.',
                        nargs='?', const='no', default='no')
    parser.add_argument('--processed_images_path',
                        help='The path and name to the tfrecord file where ' \
                             'the processed images will be saved.',
                        nargs='?', const='processed_images.tfrecord',
                        default='processed_images.tfrecord')
    ARGS = parser.parse_args()


# functions
def images_to_tfrecord(args):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    logging.info('preparing images')
    if os.path.exists(args.processed_images_path):
        logging.debug('removing file at {}'.format(
                                            args.processed_images_path))
        os.remove(args.processed_images_path)

    root_dir = Path(args.images_path)
    image_paths = list(root_dir.rglob('*.jpg'))
    '''image_paths = [os.path.join(args.images_path, image_name)
                   for image_name in os.listdir(args.images_path)]'''

    writer = tf.python_io.TFRecordWriter(args.processed_images_path)

    for image_path in tqdm(image_paths):
        image = load_image(image_path,
                           size=[configs.IMAGE_INPUT_SIZE,
                                 configs.IMAGE_INPUT_SIZE])

        if image is None:
            continue

        image = image / 255 * 2 - 1  # for hyperbolic tangent
        image = image.astype(np.float32)

        example = tf.train.Example(features=tf.train.Features(feature={
                    'name': _bytes_feature(image_path.encode()),
                    'image_raw': _bytes_feature(image.tostring())}))

        writer.write(example.SerializeToString())


def read_and_decode(filename_queue, args):
    with tf.name_scope('data'):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'name': tf.FixedLenFeature([], tf.string),
                'image_raw': tf.FixedLenFeature([], tf.string)
            })

        image_raw = tf.decode_raw(features['image_raw'], tf.float32)
        image = tf.reshape(image_raw, [configs.IMAGE_INPUT_SIZE,
                                       configs.IMAGE_INPUT_SIZE, 3])
        images = tf.train.shuffle_batch([image], batch_size=args.batch_size,
                                        capacity=5000, min_after_dequeue=1000)

    return images


def build_model(args):
    logging.info('building model')

    conv_layer = partial(tf.contrib.layers.convolution2d,
                         activation_fn=tf.nn.relu,
                         kernel_size=3,
           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    deconv_layer = partial(tf.contrib.layers.convolution2d_transpose,
                           activation_fn=tf.nn.relu,
                           kernel_size=3,
           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    fc_layer = partial(tf.contrib.layers.fully_connected,
                       activation_fn=tf.nn.relu,
                   weights_initializer=tf.contrib.layers.xavier_initializer())

    inp = tf.placeholder(tf.float32, [None, configs.IMAGE_INPUT_SIZE,
                                      configs.IMAGE_INPUT_SIZE, 3])

    conv1 = conv_layer(inp, 1024)
    conv2 = conv_layer(conv1, 512, stride=2)
    conv3 = conv_layer(conv2, 256, stride=2)
    conv4 = conv_layer(conv3, 128, stride=2)
    conv5 = conv_layer(conv4, 32, stride=2)

    conv5_flattened = tf.layers.flatten(conv5)

    bottleneck = fc_layer(conv5_flattened, round(args.bottleneck_layer_size))

    fc2 = fc_layer(bottleneck, conv5_flattened.get_shape().as_list()[1])

    conv5_shape = conv5.get_shape().as_list()
    fc2_reshaped = tf.reshape(fc2, [-1, conv5_shape[1], conv5_shape[2],
                                    conv5_shape[3]])

    deconv1 = deconv_layer(fc2_reshaped, 128, stride=2)
    deconv2 = deconv_layer(deconv1, 256, stride=2)
    deconv3 = deconv_layer(deconv2, 512, stride=2)
    deconv4 = deconv_layer(deconv3, 1024, stride=2)
    deconv5 = deconv_layer(deconv4, 3, activation_fn=tf.nn.tanh)

    tf.contrib.layers.summarize_tensors([conv1, conv2, conv3, conv4, conv5,
                                         bottleneck, fc2, deconv1, deconv2,
                                         deconv3, deconv4, deconv5])

    return inp, bottleneck, deconv5


def main():
    save_pkl_file(ARGS,
                  os.path.join(ARGS.training_path,
                               configs.METADATA_FILE_NAME))

    if not os.path.exists(ARGS.processed_images_path) or \
       ARGS.replace_img_tfrecord.lower()[0] == 'y':
        images_to_tfrecord(ARGS)

    filename_queue = tf.train.string_input_producer(
                        [ARGS.processed_images_path])

    image_batch_op = read_and_decode(filename_queue, ARGS)

    inp, bottleneck, output = build_model(ARGS)
    tf.summary.image('input', inp)
    tf.summary.image('output', output)

    logging.info('preparing session')
    with tf.name_scope('loss_mse'):
        loss = tf.reduce_mean(tf.pow(inp - output, 2))
        tf.summary.scalar('loss_mse', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(ARGS.learning_rate).minimize(loss)

    saver = tf.train.Saver(max_to_keep=ARGS.max_ckpts_to_keep)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    logging.info('starting session')
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tensorboard_path = os.path.join(ARGS.training_path,
                                        configs.TENSORBOARD_NAME)
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        logging.info('training, you can see the training progress in ' \
                     'tensorboard at {}'.format(tensorboard_path))
        step = 1
        while True:
            image_batch = sess.run(image_batch_op)
            _, summary = sess.run([optimizer, merged],
                                  feed_dict={inp: image_batch})

            train_writer.add_summary(summary, step)

            if step % ARGS.saving_interval == 0:
                _ = saver.save(sess, os.path.join(ARGS.training_path,
                                                  configs.MODEL_CHECKPOINT_NAME.format(step)))
                print('saved checkpoint at step {}'.format(step))
            if step == ARGS.end_step:
                break

            step += 1

        coord.request_stop()
        coord.join(threads)

    logging.info('finished training after exceeding {} steps'.format(
                                                              ARGS.end_step))


if __name__ == '__main__':
    main()

