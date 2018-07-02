import os, re, sys
sep = os.sep
path = os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from glob import glob
import numpy as np

def rm_end_sep(dpath):
    """去掉目录路径结尾的分隔符"""
    if dpath.endswith(sep):
        return rm_end_sep(dpath[:-1])
    else:
        return dpath

def draw_picture(src_file, pattern):
    key_list = []
    with open(src_file, 'r')  as reader:
        for line in reader:
            key = re.findall(pattern, line.strip())
            if not key: continue
            key_list.append(key[0])

    plt.plot(key_list)
    saved_file = src_file.split('.')[0]+"-loss.jpg"
    plt.savefig(saved_file)

def newest_model(save_dir, pattern):
    '''指定目录下找到最新的文件，返回文件路径和最新的epoch索引'''
    nf, ne = None, 0
    keys, fps = [], []
    for s in glob("{}{}*".format(save_dir, sep)):
        r = re.search(pattern, s)
        if r == None: continue
        keys.append(int(r.groups()[0]))
        fps.append(s)
    try:
        ne = max(keys)
        nf = fps[keys.index(ne)]
    except Exception as e:
        print(e)
    return nf, ne

def control_files_number(root_dir, pattern, num):
    '''控制某个目录下指定模式文件的数量
    参数：
        dir: 模型保存目录
        pattern: re模块需要的文件名匹配模式，用()标识排序关键词
        num: 保留文件的数量
    '''
    keys = []
    fps = []
    for s in glob("{}{}*".format(rm_end_sep(root_dir), sep)):
        r = re.search(pattern, s)
        if r == None: continue
        keys.append(int(r.groups()[0]))
        fps.append(s)
    keys = sorted(keys, reverse=True)[:num]

    for s in fps:
        kw = int(re.search(pattern, path.basename(s)).groups()[0])
        if kw not in keys: os.remove(s)

def control_array_show(arr, width=8):
    '''返回好看的数组字符串
    参数：
        arr: 列表，numpy数组，...
        element_width: 元素本身宽度
        cell_width: 单元格宽度
    '''
    arr = list(arr)
    string = ''
    for ele in arr:
        string += "{:<{}.4f}".format(ele, width)
    return string

def process_bar(i, n, pref_str='', suff_str='', char='=', num_chars=100):
        '''
        :param i: 计数器
        :param n: 计数器最大值
        :param char: 进度条符号
        :param pref_str: 进度条前置字符串
        :param suff_str: 进度条后置字符串
        '''
        i += 1
        num = i * num_chars // n
        pre = char * num
        pro = '>' + ' ' * (num_chars - 1 - num) if num - num_chars else ''
        numerator = ' ' * (len(str(n)) - len(str(i))) + str(i)
        if num - num_chars:
            sys.stdout.write("%s|%s/%d|%s%s|%s%%|%s\r" % (
            pref_str, numerator, n, pre, pro, ' ' + str(num * 100 // num_chars), suff_str))
        else:
            sys.stdout.write("%s|%s/%d|%s%s|%s%%|%s\n" % (pref_str, numerator, n, pre, pro, 100, suff_str))

def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

def calc_metrics(y_true, y_pred):
    '''计算回归结果的性能指标'''
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from math import sqrt
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

# 读取多个保存字典的文件并求交集
def add_dicts(fps, k_fn=None, v_fn=None):
    k_fn = k_fn if k_fn else lambda x: str(x)
    v_fn = v_fn if v_fn else lambda x: float(x)
    d = {}
    for fp in fps:
        with open(fp, 'r') as r:
            for row in r.read().strip().split('\n'):
                k, v = row.split(',')
                k = k_fn(k)
                v = v_fn(v)
                if d.get(k, None) == None:
                    d[k] = v
                else:
                    d[k] += v
    return d

def calc_mean(y_true, y_pred, rm_highest_number=1):
    '''Whether to remove the highest values
    Args:
        y_true: shape = (num_samples, )
        y_pred: shape = (num_models, num_samples)
    Return:
        mean
    '''
    def set_max_index_to_zero(nd_array, nd_diff):
        arr = np.array(nd_array).copy().T
        diff = np.array(nd_diff).copy().T
        index = diff.argmax(1)
        for k in range(arr.shape[0]):
            arr[k][index[k]] = 0
            diff[k][index[k]] = 0
        return arr.T, diff.T

    y_true = np.array(y_true)
    y_pred = np.array(y_pred).copy()
    (num_models, num_models) = y_pred.shape
    diff = np.abs(y_pred - y_true)

    for _ in range(rm_highest_number):
        y_pred, diff = set_max_index_to_zero(y_pred, diff)

    return y_pred.sum(0) / (num_models - rm_highest_number)

def cond_mean(y_preds, rm_top=0):

    def set_max_index_to_zero(nd_array, nd_diff):
        arr = np.array(nd_array).copy().T
        diff = np.array(nd_diff).copy().T
        index = diff.argmax(1)
        for k in range(arr.shape[0]):
            arr[k][index[k]] = 0
            diff[k][index[k]] = 0
        return arr.T

    if rm_top == 0:
        return np.array(y_preds).mean(0)

    arr = np.array(y_preds).copy()
    (num_models, num_samples) = arr.shape
    means = np.mean(arr, 0)
    diffs = np.abs(arr - means)
    arr = set_max_index_to_zero(arr, diffs)

    return arr.sum(0)/(num_models-1)

def pplot(fp):
    import pandas as pd
    df = pd.read_csv(fp, header=None, names=['true','pred'])
    df = df.sort_index(by='true')
    print(fp)
    plt.scatter(df.true, df.pred)
    plt.plot(range(100))
    plt.imsave('a.jpg')

if __name__ == '__main__':
    pplot('pp.csv')