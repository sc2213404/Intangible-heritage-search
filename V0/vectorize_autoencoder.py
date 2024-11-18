'''
Vectorize autoencoder

Program to vectorize images through taking the bottleneck layer from trained
autoencoder.

Oliver Edholm, 14 years old 2017-03-25 17:48
'''
# imports
import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tqdm import tqdm
from pathlib import Path

from autoencoder_training import build_model
from utils import configs
from utils.ops import get_pkl_file
from utils.ops import load_image
from utils.vector_file_handler import VectorSaver
from utils.vector_file_handler import establish_vectors_folder

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to vectorize ' \
                                                 'images through taking the ' \
                                                 'bottleneck layer from ' \
                                                 'trained autoencoder.')
    parser.add_argument('--training_path',
                        help='Path to folder where the training occured',
                        nargs='?', const='training', default='training')
    parser.add_argument('--vectors_path',
                        help='Path to folder where vectors will be saved.',
                        nargs='?', const='vectors', default='vectors')
    parser.add_argument('--images_path', nargs='?', const='/root/SE_WORK/shenchao/transformed_images',
                        default='/root/SE_WORK/shenchao/transformed_images', help='Path to images to vectorize.')
    ARGS = parser.parse_args()


# functions
def get_checkpoint_path(args):
    checkpoints = [name.split('.ckpt')[0] + '.ckpt'
                   for name in os.listdir(args.training_path)
                   if '.ckpt' in name]

    if checkpoints:
        return os.path.join(args.training_path, sorted(checkpoints)[-1])
    else:
        raise Exception('Couldn\'t find any checkpoint at {}'.format(
                                                        args.training_path))


def get_batches(inputs, args):
    cur_batch = []
    idx = 0
    for item in inputs:
        cur_batch.append(item)

        if (idx + 1) % args.batch_size == 0:
            yield cur_batch
            cur_batch = []

        idx += 1

    if cur_batch:
        for _ in xrange(args.batch_size - len(cur_batch)):
            cur_batch.append(np.zeros(cur_batch[0].shape))

        yield cur_batch


def main():
    training_args = get_pkl_file(os.path.join(ARGS.training_path,
                                              configs.METADATA_FILE_NAME))

    root_dir = Path(ARGS.images_path)
    original_image_paths = list(root_dir.rglob('*.jpg'))
    image_paths = list(map(str, original_image_paths))
    '''
    image_paths = [os.path.join(ARGS.images_path, file_name)
                   for file_name in os.listdir(ARGS.images_path)]
    '''

    # using generators to save memory
    images = (load_image(image_path, size=[configs.IMAGE_INPUT_SIZE,
                                           configs.IMAGE_INPUT_SIZE],
                         failure_image=np.zeros([configs.IMAGE_INPUT_SIZE,
                                                 configs.IMAGE_INPUT_SIZE, 3]))
              for image_path in image_paths)
    image_batches = get_batches(images, training_args)

    inp, bottleneck, output = build_model(training_args)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, get_checkpoint_path(ARGS))

        vectors_path = establish_vectors_folder(ARGS.vectors_path,
                                                training_args, False)
        vector_saver = VectorSaver(vectors_path)

        length = np.floor(len(image_paths) / training_args.batch_size) + \
                 int(bool(len(image_paths) / training_args.batch_size))
        idx = 0
        logging.info('vectorizing')
        for image_batch in tqdm(image_batches, total=length):
            vectors = sess.run(bottleneck, feed_dict={inp: image_batch})

            for vector in vectors:
                vector_saver.add_vector(image_paths[idx], vector)

                idx += 1
                if idx == len(image_paths):
                    break

    logging.info('saved data at {}'.format(vectors_path))


if __name__ == '__main__':
    main()
