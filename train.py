import os, argparse, glob, sys, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sep = os.sep
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from utils import rm_end_sep
from utils import control_files_number
from utils import control_array_show
from utils import newest_model
from utils import draw_picture
from utils import add_dicts
from model import read_tfr
from model import model
parse = argparse.ArgumentParser()
parse.add_argument('-d')
parse.add_argument('-bs', type=int, default=32)
parse.add_argument('-lr', type=float, default=0.00005)
parse.add_argument('-dropout', type=float, default=0.5)
parse.add_argument('-epoch', type=int, default=3000)
parse.add_argument("-gpu")
parse.add_argument("-ckpt", action="store_true")
ps = parse.parse_args()
if ps.gpu == None: raise Exception("no gpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ps.gpu

DROPOUT = ps.dropout
NUM_EPOCHES = ps.epoch
MODEL_DIR = rm_end_sep(ps.d)
SEARCH_PATTERN = "(\d+)\.npz"
TBLOG_DIR = MODEL_DIR + os.sep + "logs"

src_dir = "/home/chenyin/dataDir/"
TRAIN_FILES = [
    src_dir + "/output-0/train.tfr",
    src_dir + "/output-0/train-filters.tfr",
    src_dir + "/output-1/train.tfr",
    src_dir + "/output-1/train-filters.tfr",
    # res-4
    # src_dir + "age-filter-train.tfr",
]
EVAL_FILES = [
    src_dir+ "/output-0/eval.tfr",
    src_dir+ "/output-1/eval.tfr",
]
print('[*] Load samples info ...')
info = add_dicts(
    fps=[
        "/home/chenyin/dataDir/output-0/info.txt",
        "/home/chenyin/dataDir/output-0/info-filters.txt",
        "/home/chenyin/dataDir/output-1/info.txt",
        "/home/chenyin/dataDir/output-1/info-filters.txt",
    ],
    k_fn=lambda x: str(x),
    v_fn=lambda x: int(x)
)
print(
    "[*] samples info:\n" + \
    "--- num total: {}\n".format(info['N']) + \
    "--- num train: {} => {}\n".format(info['NTRaw'], info['NT']) + \
    "--- num eval: {}".format(info['NE'])
)

# NUM_TRAIN_SAMPLES = info['NT']
NUM_TRAIN_SAMPLES = 365
NUM_EVAL_SAMPLES = info['NE']
NUM_BATCHES = NUM_TRAIN_SAMPLES // ps.bs
MAX_MODEL_FILES = 10

def train():

    res = model(True, lr=ps.lr, show_layers_info=False, dropout=DROPOUT)
    imgs_t, labels_t = tf.train.shuffle_batch(read_tfr(TRAIN_FILES), ps.bs, 200, 50)
    imgs_e, labels_e = tf.train.shuffle_batch(read_tfr(EVAL_FILES), 6, 200, 50)
    merged_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer1 = tf.summary.FileWriter(TBLOG_DIR+'-train', sess.graph)
        writer2 = tf.summary.FileWriter(TBLOG_DIR+'-eval', sess.graph)
        (nf, ne) = newest_model(MODEL_DIR, SEARCH_PATTERN) if ps.ckpt else (None, 0)
        if nf != None:
            print("[*] Checkpoint and continue train ...")
            tl.files.load_and_assign_npz(sess, nf, res['net'])

        try:
            print("[*] Start training ...")
            for eidx in range(ne+1, NUM_EPOCHES+1):
                for _ in range(NUM_BATCHES):
                    X_t, y_t = sess.run([imgs_t, labels_t])
                    fetches_t = [res['train_op'], res['loss'], res['prediction'], res['gstep'], merged_summary]
                    feed_dict = {res['x']: X_t, res['y_real']: y_t}
                    feed_dict.update(res['net'].all_drop)
                    r_t = sess.run(fetches_t, feed_dict)

                ## Evaluate
                if eidx % 5 == 0:
                    writer1.add_summary(r_t[4], eidx)

                    X_e, y_e = sess.run([imgs_e, labels_e])
                    feed_dict_e = {res['x']:X_e, res['y_real']:y_e}
                    feed_dict_e.update(tl.utils.dict_to_one(res['net'].all_drop))

                    fetches_e = [res['loss'], res['prediction'], merged_summary]
                    r_e = sess.run(fetches_e, feed_dict=feed_dict_e)

                    print("epoch={}, loss={}, eval_loss={}".format(eidx, r_t[1], r_e[0]))
                    writer2.add_summary(r_e[2], eidx)

                ## Save parameters
                if eidx % 20 == 0:
                    name = "{}{}{}.npz".format(MODEL_DIR, sep, eidx)
                    # Control number of files with model parameters saved.
                    control_files_number(MODEL_DIR, SEARCH_PATTERN, MAX_MODEL_FILES)
                    tl.files.save_npz(res['net'].all_params, name=name)

            coord.request_stop()
            coord.join(threads=threads)

        except KeyboardInterrupt:
            name = "{}{}{}.npz".format(MODEL_DIR, sep, eidx)
            tl.files.save_npz(res['net'].all_params, name=name)
            print("[*] Save params to {}".format(name))

if __name__ == '__main__':
    if ps.ckpt == False:
        print("[*] Remove and remake directory: {}".format(MODEL_DIR))
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.mkdir(MODEL_DIR)
    train()
