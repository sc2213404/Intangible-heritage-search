# imports
import argparse
import logging
import os
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from six.moves import xrange
from pathlib import Path

from autoencoder_training import build_model
from utils import configs
from utils.ops import get_pkl_file
from utils.ops import load_image
from utils.vector_file_handler import VectorLoader
from vectorize_autoencoder import get_checkpoint_path
from vectorize_pretrained import build_graph
from vectorize_pretrained import get_size

slim = tf.contrib.slim

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
#if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Program evaluating vectors ' \
                                                'created.')
# parser.add_argument('--image_path', help='Path to image to evaluate ', default="static\\images\\test.jpg")
parser.add_argument('--vectors_path', help='Path to folder where metadata ' \
                                        'and vectors are saved.',
                    default='vectors\\vectors_1')
    
parser.add_argument('--similarity_func',
                    help='Which distance function to use to get distance ' \
                        'between two vectors. Functions currently ' \
                        'available are: cosine and euclidean.', nargs='?',
                    const='cosine', default='cosine')
ARGS = parser.parse_args()


# functions
def get_similarity_func(name):
    name = name.lower()
    if name in ['cosine', 'cos']:
        return cosine
    elif name in ['euclidean', 'euc']:
        return euclidean
    else:
        raise 'Unknown distance function: {}'.format(name)


def load_vector_data(vector_dir_path):
    vector_generator = VectorLoader(vector_dir_path).get_vectors_generator()
    args = get_pkl_file(os.path.join(vector_dir_path,
                                     configs.METADATA_FILE_NAME))
    with open(os.path.join(vector_dir_path, configs.TYPE_FILE_NAME)) \
            as txt_file:
        vector_type = txt_file.read()

    return vector_generator, args, vector_type


def get_autoencoder_vector(image_path, args):
    image = load_image(image_path, size=[configs.IMAGE_INPUT_SIZE,
                                         configs.IMAGE_INPUT_SIZE])
    batch = [image]
    for _ in xrange(args.batch_size - 1):
        batch.append(np.zeros(image.shape))

    inp, bottleneck, output = build_model(args)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, get_checkpoint_path(args))
        vectors = sess.run(bottleneck, feed_dict={inp: batch})
        return list(vectors)[0]


def get_pretrained_vector(image_path, args):
    size = get_size(args)
    image = load_image(image_path, size=[size, size])
    batch = [image]
    for _ in xrange(configs.BATCH_SIZE - 1):
        batch.append(np.zeros(image.shape))  # arbitrary size

    vectorize_op, inps_placeholder = build_graph(args)

    init = tf.global_variables_initializer()
    init_fn = slim.assign_from_checkpoint_fn(args.model_path,
                                             slim.get_model_variables())
    with tf.Session() as sess:
        sess.run(init)
        init_fn(sess)

        vectors = sess.run(vectorize_op,
                           feed_dict=dict(zip(inps_placeholder, batch)))
        return vectors[0]


def main(img_path):
    vector_generator, args, vector_type = load_vector_data(ARGS.vectors_path)

    if vector_type == 'pretrained':
        image_vector = get_pretrained_vector(img_path, args)
    elif vector_type == 'autoencoder':
        image_vector = get_autoencoder_vector(img_path, args)
    else:
        raise Exception('Unknown vector type: {}'.format(vector_type))

    if len(image_vector.shape) != 1:
        image_vector = image_vector.flatten()

    similarity_func = get_similarity_func(ARGS.similarity_func)

    logging.info('getting closest vector')
    closest_vector_name = None
    closest_dist = float('inf')
    lst_dist = []
    for name, vector in vector_generator:
        dist = similarity_func(image_vector, vector)
        lst_dist.append((dist, name))

    lst_dist.sort(key = lambda x: x[0])
    lst_path = []
    lst_name = []
    '''
    for _, name in lst_dist:
        path = Path(name)
        grandparent_dir = path.parent
        if grandparent_dir in lst_path:
            continue
        else:
            lst_path.append(grandparent_dir)
            lst_name.append(str(grandparent_dir) + '\\1.jpg')
        if len(lst_name) == 100:
            break
    '''
    count = 0
    for _, name in lst_dist:
        path = Path(name)
        lst_name.append(path)
        count += 1
        if count == 7:
            break

    for name, vector in vector_generator:
        dist = similarity_func(image_vector, vector)

        if dist < closest_dist:
            closest_dist = dist
            closest_vector_name = name


    print('most similar image to {} is {}'.format(
        img_path, lst_name))

    # 获取父文件夹名称
    file_name = os.path.basename(os.path.dirname(lst_name[0]))
    '''
    # 读取并展示图片
    for item in lst_name:
        img = mpimg.imread(item)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()
    '''
    return lst_name, file_name

'''
if __name__ == '__main__':
    start = time.time()
    lst, name = main('1.jpg')
    end = time.time()
    print(end - start)
    print(lst)
    print(name)
'''