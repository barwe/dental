# coding=utf-8
import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from sys import argv
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorlayer.files import load_and_assign_npz
from model import model
from model import read_tfr
from utils import calc_metrics, add_dicts, cond_mean
from build_tfr import TFRecordFileBuilder
ps = ArgumentParser()
ps.add_argument('-m', help="model dir")
ps.add_argument('-p', default="\d+\.npz", help="model file pattern")
# ps.add_argument('-d', help="imgs dir for testing")
# ps.add_argument('-tfr', default="/home/chenyin/dataDir/dentale_data/raw-data/eval.tfr")
# ps.add_argument('-tfr-info', default="/home/chenyin/dataDir/dentale_data/raw-data/info.txt")
# ps.add_argument('-tfr-info-key', default="num_eval_samples")
ps.add_argument('-o', '--output')
ps.add_argument('-rm-top', type=int, default=0)
ps = ps.parse_args()
#
EVAL_FILES = [
    "/home/chenyin/dataDir//output-0/eval.tfr",
    "/home/chenyin/dataDir//output-1/eval.tfr",
]
INFO_FILES = [
    "/home/chenyin/dataDir/output-0/info.txt",
    "/home/chenyin/dataDir/output-1/info.txt",
]



# need ps.tfr_info, ps.tfr_info_key, ps.tfr
def load_data_from_tfr():
    print("[*] Load TFRecord file")
    info = add_dicts(
        fps=INFO_FILES,
        k_fn=lambda x: str(x),
        v_fn=lambda x: int(x)
    )
    X, y_true = tf.train.shuffle_batch(read_tfr(EVAL_FILES), info['NE'], info['NE'], 0)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        X, y_true = sess.run([X, y_true])

    return {'X': X, 'y': y_true, 'num_samples': info['NE']}

# need ps.m, ps.p
def get_model_list():
    model_dir = ps.m
    pattern = ps.p
    if model_dir.endswith(os.sep):
        model_dir = model_dir[:-1]

    models = []
    for item in glob("{}{}*".format(model_dir, os.sep)):
        results = re.findall(pattern, item)
        if results:
            models.append("{}{}{}".format(model_dir, os.sep, results[0]))
    return models

# need ps.output
def pred():
    batch_size = 32
    output_dir = ps.output
    output_dir = output_dir[:-1] if output_dir.endswith(os.sep) else output_dir
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    rm_top = ps.rm_top

    data = load_data_from_tfr()
    num_samples = data['num_samples']
    num_batches = num_samples // batch_size
    batch_left = num_samples % batch_size
    if batch_left != 0: num_batches += 1

    models = get_model_list()
    num_models = len(models)
    res = model(is_train=False, show_layers_info=False)
    predictions = np.zeros((num_models, num_samples)) #(5, 80)
    for idx in range(num_models):
        sess = None
        with tf.Session() as sess:
            load_and_assign_npz(sess, models[idx], res['net'])
            for batchIdx in range(num_batches):
                if batchIdx == num_batches-1:
                    batch_length = batch_left if batch_left else batch_size
                    batch_X = data['X'][batchIdx*batch_size:]
                else:
                    batch_length = batch_size
                    batch_X = data['X'][batchIdx*batch_size:(batchIdx+1)*batch_size]
                prediction = sess.run([res['prediction']], feed_dict={res['x']: batch_X})
                predictions[idx, batchIdx*batch_size:batchIdx*batch_size+batch_length] = np.squeeze(prediction)

    # predictions = np.mean(predictions, axis=0)
    predictions = cond_mean(predictions, rm_top=rm_top)
    predictions = predictions * 100

    TFRecordFileBuilder._write_dict(
        "{}{}metrics-results.csv".format(output_dir, os.sep),
        calc_metrics(data['y'], predictions)
    )

    write2file(
        "{}{}predictions-results.csv".format(output_dir, os.sep),
        data['y'],
        predictions
    )

def write2file(output_file, y_true, y_pred):
    assert len(y_true) == len(y_pred), ""
    string = '\n'.join(["{},{}".format(x, y) for x, y in zip(y_true, y_pred)])
    with open(output_file, 'w') as writer:
        writer.write(string)
    print("[*] Write predictions to {}".format(output_file))



pred()
# a=np.array([[1,4],[3,3],[4,6],[8,3]])
# print('a',a  )
# b=cond_mean(a)
# print(b)