import os, re, sys, glob, xlrd, shutil, random
sep = os.sep
path = os.path
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import rm_end_sep
from utils import process_bar
from utils import is_number

class TFRecordFileBuilder(object):

    def __init__(self,
                 std_shape=(447,447),
                 train_ratio=0.9,
                 cut_shape=(1260,2420),
                 uniform_shape=(1300,2500),
                 slid_strides=(40,80),
                 verbose=0,
                 ):

        self.train_ratio = train_ratio
        # 日志打印控制
        self.verbose = verbose
        # 网络标准输入尺寸
        self.std_shape = std_shape
        # 训练时滑动增强强对图像尺寸进行统一
        self.uniform_shape = uniform_shape
        # 训练时滑动增强窗口大小，测试时剪裁窗口大小
        self.cut_shape = cut_shape
        # 训练时滑动步长
        self.slid_strides = slid_strides

    @classmethod # auto-called.
    def _write_tfr(cls, writer, image, label):

        image_fn = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        label_fn = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": label_fn,
                    "image": image_fn
                }
            )
        )
        writer.write(example.SerializeToString())

    @classmethod # write dict content into file.
    def _write_dict(cls, fp, D):
        assert isinstance(D, dict), ''
        with open(fp, 'w') as writer:
            string = '\n'.join(["{},{}".format(k, v) for k, v in D.items()])
            writer.write(string)
        print("[*] Make {} SUCCESS!".format(fp))

    @classmethod # load content into dict object.
    def _read_dict(cls, fp, k_trans_fn=None, v_trans_fn=None):
        k_trans_fn = k_trans_fn if k_trans_fn else lambda x: x
        v_trans_fn = v_trans_fn if v_trans_fn else lambda x: x
        return_dict = {}
        with open(fp, 'r') as r:
            for row in r.read().strip('\n').split('\n'):
                k, v = row.split(',')
                k = k_trans_fn(k)
                v = v_trans_fn(v)
                return_dict[k] = v
        return return_dict

    @classmethod # write list content into file.
    def _write_list(cls, fp, L):
        assert isinstance(L, list), ''
        with open(fp, 'w')  as w:
            string = '\n'.join(L)
            w.write(string)
        print("[*] Make {} SUCCESS!".format(fp))

    @classmethod # load content into list object.
    def _read_list(cls, fp):
        assert os.path.exists(fp)
        with open(fp, 'r') as r:
            return r.read().strip('\n').split('\n')

    @classmethod
    def _get_crop_box(cls, img, shape):
        """返回图片居中的大小为shape的box供crop使用
        Args:   img: 灰度图（2D）
        """
        img_arr = np.array(img)
        imgH, imgW = img_arr.shape
        cropTop = (imgH - shape[0]) // 2
        cropLeft = (imgW - shape[1]) // 2
        return (cropLeft, cropTop, cropLeft+shape[1], cropTop+shape[0])

    def _slid_and_tfr(self, writer, img, label, **kwargs):
        """给定原始图片进行滑动并且保存为TFRecord"""
        slid_window = kwargs['slid_window']
        slid_stride = kwargs['slid_stride']
        resize_after_slid = kwargs['resize_after_slid']
        filters = kwargs.get('image_filters')
        fsize = img.size  # (2500,1300)
        N = 0 # 子图数量
        for h in range(0, fsize[0] - slid_window[0] + 1, slid_stride[0]):
            for v in range(0, fsize[1] - slid_window[1] + 1, slid_stride[1]):
                sub_img = img.crop(box=(h, v, slid_window[0] + h, slid_window[1] + v))
                if resize_after_slid != None:
                    sub_img = sub_img.resize(size=resize_after_slid)
                # self._write_tfr(writer, sub_img, label)
                N += self.image_filter(writer, sub_img, label, filters)
        return N

    @classmethod
    def load_xlsxs(cls, fps, k_idx, k_fn, v_idx, v_fn):

        def load_xlsx(fp, k_idx, k_fn, v_idx, v_fn):
            labels = {}
            for sheet in xlrd.open_workbook(filename=fp).sheets():
                for sampleIdx in range(sheet.nrows):
                    splitSample = sheet.row_values(sampleIdx)
                    key, value = splitSample[k_idx], splitSample[v_idx]
                    # if not value or not is_number(value):
                    if not value:
                        continue
                    try:
                        key = k_fn(key)
                        value = v_fn(value)
                    except Exception:
                        continue
                    labels[key] = value
            return labels

        if isinstance(fps, str):
            fps = [fps,]
        D = {}
        for fp in fps:
            if not os.path.exists(fp): continue
            D.update(load_xlsx(fp, k_idx, k_fn, v_idx, v_fn))
        return D

    def _load_fps(self, dp, ft='jpg'):
        """
        :param dp: dir_path
        :param ft: file_type
        :return:
        """
        return glob.glob(dp+os.sep+"*."+ft)

    def _map_excel_to_fps(self, D, L, fn):
        """
        :param D: 文件关键字=>标签的字典
        :param L: 文件路径列表
        :param fn: 映射函数
        :return: 文件路径=>标签的字典
        """
        newD = {}
        for fp in L:
            k = fn(os.path.basename(fp))
            if D.get(k, 'NE') == 'NE':
                continue
            else:
                newD[fp] = D[k]
        return newD

    def _make_te_set(self, fps, t_rate):
        random.shuffle(fps)
        N = len(fps)
        NT = int(N * t_rate)
        t_set = fps[:NT]
        e_set = fps[NT:]
        return {
            't_set': t_set,
            'e_set': e_set,
            't_num': NT,
            'e_num': N - NT,
            'num': N
        }

    def _make_t_tfr(self, T_set, D, T_tfr, **kwargs):
        """
        :param T_set:
        :param D: 文件路径-标签字典
        :param T_tfr:
        :return:
        """
        filters = kwargs.get('image_filters')
        N = 0 # 子图数量
        with tf.python_io.TFRecordWriter(T_tfr) as w:
            num_t = len(T_set)
            for idx in range(num_t):
                process_bar(idx, num_t, "[*] Make T_tfr ", '', '=', 25)
                fp = T_set[idx]
                label = D.get(fp, "NE")
                if label == "NE":
                    continue
                img = Image.open(fp).convert('L')
                ## 剪切以统一尺寸
                crop_box = self._get_crop_box(img, shape=self.uniform_shape)
                img_1 = img.crop(crop_box)
                # 滑动增强的同时保存到TFRecord
                N += self._slid_and_tfr(w, img_1, label,
                                        slid_window=(self.cut_shape[1], self.cut_shape[0]),
                                        slid_stride=(self.slid_strides[1], self.slid_strides[0]),
                                        resize_after_slid=self.std_shape,
                                        image_filters=filters,
                                        )
                # 正中间取一张
                img_2 = img.crop(self._get_crop_box(img, shape=self.cut_shape))
                img_2 = img_2.resize(size=self.std_shape)
                # self._write_tfr(w, img_2, label)
                N += self.image_filter(w, img_2, label, filters)
        return N

    def _make_e_tfr(self, E_set, D, E_tfr):
        with tf.python_io.TFRecordWriter(E_tfr) as w:
            num_e = len(E_set)
            for idx in range(num_e):
                process_bar(idx, num_e, "[*] Make E_tfr ", '', '=', 25)
                fp = E_set[idx]
                label = D.get(fp, "NE")
                if label == "NE": continue
                img = Image.open(fp).convert('L')
                ## 剪切以统一尺寸
                crop_box = self._get_crop_box(img, shape=self.cut_shape)
                img = img.crop(crop_box)
                ## 缩放保存
                img = img.resize(size=self.std_shape)
                self._write_tfr(w, img, label)
        return num_e

    @classmethod
    def image_filter(cls, TFR_writer, image, label, filters=None):
        """对子图进行滤镜增强，写入tfr文件
        返回子图生成的子图数N：无滤镜时为1，否则为滤镜数量"""
        if filters == None:
            cls._write_tfr(TFR_writer, image, label)
            return 1
        else:
            for item in filters:
                sub_img = image.filter(item)
                cls._write_tfr(TFR_writer, sub_img, label)
            return len(filters)

    def _load_set_and_dict(self, fps_file, kv_file, **kwargs):
        k_col_idx = kwargs.get('k_col_idx', 0)  # k所在的列索引
        v_col_idx = kwargs.get('v_col_idx', 2)  # v所在的列索引
        # 从excel文件加载k时的转换函数
        k_trans_fn = kwargs.get('k_trans_fn', lambda x: x if isinstance(x, str) else str(int(x)))
        # 从excel文件加载v时的转换函数
        v_trans_fn = kwargs.get('v_trans_fn', lambda x: float(x))
        # 将从excel加载而来的字典映射到文件路径列表
        map_fn = kwargs.get('map_fn', lambda x: re.findall(r'([^.]+)\.jpg', x)[0])

        t_set = self._read_list(fps_file)
        data_dict = self.load_xlsxs(kv_file, k_col_idx, k_trans_fn, v_col_idx, v_trans_fn)
        data_dict = self._map_excel_to_fps(
            D=data_dict,
            L=t_set,
            fn=map_fn
        )

        return t_set, data_dict

    #entry:访问原始图像目录对数据集进行切割并构建训练tfr和测试tfr，仅有滑动增强
    def make_TFR_file(self, **kwargs):

        img_dir = kwargs['img_dir'] # 文件所在目录，无子目录
        kv_file = kwargs['kv_file'] # 这里为保存kv信息的excel文件
        k_col_idx = kwargs['k_col_idx'] # k所在的列索引
        v_col_idx = kwargs['v_col_idx'] # v所在的列索引
        k_trans_fn = kwargs['k_trans_fn'] # 从excel文件加载k时的转换函数
        v_trans_fn = kwargs['v_trans_fn'] # 从excel文件加载v时的转换函数
        map_fn = kwargs['map_fn'] # 将从excel加载而来的字典映射到文件路径列表

        # 所有的输出文件
        output_dir = kwargs['output_dir']
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        train_tfr = output_dir + os.sep + 'train.tfr'
        train_txt = output_dir + os.sep + 'train.txt'
        eval_tfr = output_dir + os.sep + 'eval.tfr'
        eval_txt = output_dir + os.sep + 'eval.txt'
        info_txt = output_dir + os.sep + 'info.txt'

        # 所有图片文件路径的列表
        valid_fps = self._load_fps(img_dir, ft='jpg')

        # 数据集切割信息(或者从已有列表加载）
        sets = self._make_te_set(valid_fps, t_rate=self.train_ratio)

        # 加载kv字典：从文件加载所有
        data_dict = self.load_xlsxs(kv_file, k_col_idx, k_trans_fn, v_col_idx, v_trans_fn)
        # 过滤kv字典：基于valid_fps进行过滤
        data_dict = self._map_excel_to_fps(data_dict, valid_fps, map_fn)

        # 构建TFR文件
        Nt = self._make_t_tfr(sets['t_set'], data_dict, train_tfr)
        Ne = self._make_e_tfr(sets['e_set'], data_dict, eval_tfr)

        # 将切割好的训练集和测试集写入文件
        self._write_list(train_txt, sets['t_set'])
        self._write_list(eval_txt, sets['e_set'])
        self._write_dict(info_txt, {
            'N': sets['num'], # 原始样本总数
            'NTRaw': sets['t_num'], # 原始训练集总数
            'NERaw': sets['e_num'], # 原始测试集总数
            'NT': Nt, # 增强训练集总数
            'NE': Ne,
        })

    #entry:加载fps文件（存储用于训练的文件路径）进行滑动增强和滤镜增强后保存为tfr文件
    def load_fps_and_augment_by_filters(self, **kwargs):

        fps_file = kwargs['fps_file'] # 存有文件名路径的文件
        output_file = kwargs['output_file'] # 生成的tfr文件
        image_filters = kwargs['image_filters'] # 图像滤镜列表
        kv_file = kwargs['kv_file']  # 这里为保存kv信息的excel文件

        t_set, data_dict = self._load_set_and_dict(fps_file, kv_file, kwargs)
        N = self._make_t_tfr(t_set, data_dict, output_file, image_filters=image_filters)
        with open("{}{}info-filters.txt".format(os.path.dirname(output_file), os.sep), 'w') as w:
            w.write("NT,{}".format(N))

    # entry:加载fps文件进行年龄段的筛选构建tfr文件
    def loadFps_and_filterByAge_and_makeTfr(self, **kwargs):
        fps_files = kwargs['fps_files']  # 存有文件名路径的文件
        kv_file = kwargs['kv_file']  # 这里为保存kv信息的excel文件
        age_section = kwargs['age_section'] # 限制年龄区间
        output_file = kwargs['output_file']  # 生成的tfr文件

        t_set, data_dict = [], {}
        for fp in fps_files:
            _t_set, _data_dict = self._load_set_and_dict(fp, kv_file)
            t_set.extend(_t_set)
            data_dict.update(_data_dict)
        new_dict = {}
        for k, v in data_dict.items():
            if age_section[0] < v < age_section[1]:
                new_dict[k] = data_dict[k]
        # for k in new_dict: print(k, new_dict[k])
        t_set = list(new_dict.keys())
        N = self._make_t_tfr(t_set, data_dict, output_file)
        print("[*] Num samples: {}".format(N))

    # entry:查看训练数据的分布情况
    def show_dist(self, fps_files, kv_file, output_file):
        from matplotlib import pyplot as plt
        data_dict = {}
        for fp in fps_files:
            print("[*] Load {}".format(fp))
            _, _data_dict = self._load_set_and_dict(fp, kv_file)
            data_dict.update(_data_dict)

        values = list(data_dict.values())
        plt.hist(values, bins=np.arange(0,101,5))
        plt.savefig(output_file)
        print("[*] Num points: {}".format(len(values)))
        print("[*] Save to {}".format(output_file))

    def stat(self, **kwargs):
        root = "/home/chenyin/dataDir/"
        num_dirs = 2
        t_fps = ["{}output-{}/train.txt".format(root, i) for i in range(num_dirs)]
        e_fps = ["{}output-{}/eval.txt".format(root, i) for i in range(num_dirs)]
        kv_file = "/home/chenyin/dataDir/all_labels.xlsx"

        v_idx = 1
        v_fn = lambda x: str(x)

        t_s_1, t_d_1 = self._load_set_and_dict(t_fps[0], kv_file, v_col_idx=v_idx, v_trans_fn=v_fn)
        t_s_2, t_d_2 = self._load_set_and_dict(t_fps[1], kv_file, v_col_idx=v_idx, v_trans_fn=v_fn)
        e_s_1, e_d_1 = self._load_set_and_dict(e_fps[0], kv_file, v_col_idx=v_idx, v_trans_fn=v_fn)
        e_s_2, e_d_2 = self._load_set_and_dict(e_fps[1], kv_file, v_col_idx=v_idx, v_trans_fn=v_fn)

        t_s = t_s_1 + t_s_2
        e_s = e_s_1 + e_s_2
        t_d = {}
        t_d.update(t_d_1)
        t_d.update(t_d_2)
        e_d = {}
        e_d.update(e_d_1)
        e_d.update(e_d_2)

        # 训练集
        d1 = {'m':0, 'f':0, 'n':0}
        for k, v in t_d.items():
            if v == 'm': d1['m'] += 1
            if v == 'f': d1['f'] += 1
            else: d1['n'] += 1
        # 测试集
        d2 = {'m':0, 'f':0, 'n':0}
        for k, v in e_d.items():
            if v == 'm': d2['m'] += 1
            if v == 'f': d2['f'] += 1
            else: d2['n'] += 1

        print(d1)
        print(d2)

# 读取储存fp-v的csv文件并完成
def preprocess_by_read_csv(csv_fp, output_dir, slid_window, std_shape):
    """
    :param csv_fp: 存储文件路径-标签的csv文件
    :param output_dir: 文件输出目录
    :param slid_window: 滑动窗口的尺寸
    :param std_shape: 网络输入尺寸
    :return: 执行状态
    """
    # 输出目录安全性检查
    if os.path.exists(output_dir):
        if glob.glob("{}{}*".format(output_dir, os.sep)):
            raise Exception("[!] 目录不为空，无法进行操作：{}".format(output_dir))
        else:
            os.rmdir(output_dir)
    os.makedirs(output_dir)

    with open(csv_fp, 'r') as reader:
        fps = reader.read().strip().split('\n')
        fps_len = len(fps)
        for idx in range(fps_len):
            process_bar(idx, fps_len, '[*] Crop and resize ', '', '=', 25)
            fp = fps[idx]
            img = Image.open(fp).convert('L')
            crop_box = TFRecordFileBuilder._get_crop_box(img, shape=slid_window)
            img = img.crop(crop_box)
            img = img.resize(size=std_shape)
            new_fp = "{}{}{}".format(output_dir, os.sep, os.path.basename(fp))
            img.save(new_fp)
    print("[*] Crop and resize SUCCESS!\n--- input-file: {}\n--- output-dir: {}".format(csv_fp, output_dir))

from PIL import ImageFilter

## 实例函数
def main(rfb):
    root = "/home/chenyin/dataDir/"
    root_dirs = [
        "/home/chenyin/dataDir/raw/",
        "/home/chenyin/dataDir/zl/"
    ]
    kv_files = [
        "/home/chenyin/dataDir/all_labels.xlsx",
        "/home/chenyin/dataDir/all_labels.xlsx",
    ]

    # for idx in range(len(root_dirs)):
    for idx in range(0):
        rfb.make_TFR_file(
            img_dir=root_dirs[idx],
            kv_file=kv_files[idx],
            output_dir="{}output-{}".format(root, idx),
            k_col_idx=0,
            v_col_idx=2,
            k_trans_fn=lambda x: x if isinstance(x, str) else str(int(x)),
            v_trans_fn=lambda x: float(x),
            map_fn=lambda x: re.findall(r'([^.]+)\.jpg', x)[0],
        )

    for idx in range(len(root_dirs)):
        rfb.load_fps_and_augment_by_filters(
            fps_file="{}output-{}/train.txt".format(root, idx),
            output_file="{}output-{}/train-filters.tfr".format(root, idx),
            image_filters=[
                ImageFilter.BLUR,
                ImageFilter.EDGE_ENHANCE,
                ImageFilter.DETAIL
            ],
            kv_file=kv_files[idx],
            k_col_idx=0,
            v_col_idx=2,
            k_trans_fn=lambda x: x if isinstance(x, str) else str(int(x)),
            v_trans_fn=lambda x: float(x),
            map_fn=lambda x: re.findall(r'([^.]+)\.jpg', x)[0],
        )


if __name__ == '__main__':

    tfrfb = TFRecordFileBuilder()
    # main(tfrfb)
    # tfrfb.loadFps_and_filterByAge_and_makeTfr(
    #     fps_files=[
    #         "/home/chenyin/dataDir/output-0/train.txt",
    #         "/home/chenyin/dataDir/output-1/train.txt",
    #     ],
    #     kv_file="/home/chenyin/dataDir/all_labels.xlsx",
    #     output_file="/home/chenyin/dataDir/age-filter.tfr",
    #     age_section=(40,100)
    # )
    # sys.exit()
    # tfrfb.show_dist(
    #     fps_files=[
    #         "/home/chenyin/dataDir/output-0/train.txt",
    #         "/home/chenyin/dataDir/output-1/train.txt",
    #     ],
    #     kv_file="/home/chenyin/dataDir/all_labels.xlsx",
    #     output_file="/home/chenyin/project/tooth3/temp/c.jpg"
    # )
    # tfrfb.stat()