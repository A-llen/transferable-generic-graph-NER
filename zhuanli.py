#! -*- coding: utf-8 -*-
# 数据集 https://github.com/CLUEbenchmark/CLUENER2020

# 这是python2.7那个笔记本的，但是环境崩了，不能用，现在不需要
# import sys   #reload()之前必须要引入模块
# reload(sys)
# sys.setdefaultencoding('utf-8')

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from layers import Zhuanli
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from keras.layers import Dropout
from tqdm import tqdm
import tensorflow as tf
from keras.losses import binary_crossentropy

maxlen = 256
epochs = 8
fine_tune_epoches = 5
batch_size = 16
learning_rate = 2e-5
categories = set()
# a = {
#     '地址': 'address',
#     '书名': 'book',
#     '公司': 'company',
#     '游戏': 'game',
#     '政府': 'government',
#     '电影': 'movie',
#     '姓名': 'name',
#     '组织机构': 'organization',
#     '职位': 'position',
#     '景点': 'scene',
# }
categories_to_chinese = {
    'address': '地址',
    'book': '书名',
    'company': '公司',
    'game': '游戏',
    'government': '政府',
    'movie': '电影',
    'name': '姓名',
    'organization': '组织机构',
    'position': '职位',
    'scene': '景点',
    # 'animal': '动物',
}

# bert配置
config_path = r'bert/bert_config.json'
checkpoint_path = r'bert/bert_model.ckpt'
dict_path = r'bert/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(filename):
    """加载数据
    单条格式：[text, class, (start, end), (start, end), ...]
    """
    D = []
    all_classes = list(categories_to_chinese.keys())
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            included_classes = []
            for k, v in l['label'].items():
                categories.add(k)
                included_classes.append(k)
                d = [l['text'], k]
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end))
                D.append(d)
            # 生成本条文本不包含的类别的训练数据，但是啊！！！这个后面算准确率啥的怎么算
            rest = [item for item in all_classes if item not in included_classes]
            for c in rest:
                d = [l['text'], c]
                D.append(d)
    return D



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)  # 加入[CLS][SEP]
            chinese_class_to_append = categories_to_chinese[d[1]]
            chinese_class_tokens = tokenizer.tokenize(chinese_class_to_append, maxlen=maxlen)[1:-1]
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            tokens.extend(chinese_class_tokens)
            token_ids = tokenizer.tokens_to_ids(tokens)
            Len = len(token_ids)-len(chinese_class_tokens)
            segment_ids = [0] * (Len)+[1]*len(chinese_class_tokens)
            labels = np.zeros((maxlen, maxlen))
            for start, end in d[2:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    last = 0
                    for i in range(start, end+1, 1):
                        labels[last, i] = 1
                        last = i
                    labels[end, Len-1] = 1
            # R = []
            # decode_entities(labels[:Len, :Len], 0, Len, R)
            # print(R)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids, value=1)  # 修改segment_ids后面padding为1
                batch_labels = sequence_padding(batch_labels, seq_dims=2)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# # 调试查看数据
# train_generator = data_generator(train_data, batch_size)
# data_ge = train_generator.__iter__()
# embeding_and_segments, labels = next(data_ge)
# embeding = embeding_and_segments[0]
# segments = embeding_and_segments[1]
# print(len(embeding), embeding)
# print(len(segments), segments)
# print(len(labels), labels)


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def precision(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
    upper_tri = tf.linalg.band_part(K.ones_like(y_pred), 0, -1)  # 上三角全1，包括主对角线
    duijiaoxian = tf.linalg.band_part(K.ones_like(y_pred), 0, 0)  # 主对角线
    mask = upper_tri - duijiaoxian  # 上三角，不包括对角线
    # y_true = K.print_tensor(y_true, message='\ny_true = ')
    # y_pred = K.print_tensor(y_pred, message='\ny_pred = ')
    tensor_equ = K.cast(K.equal(y_true, y_pred), K.floatx())
    # tensor_equ = K.print_tensor(tensor_equ, message='\n相等矩阵 ')
    return K.sum(tensor_equ*mask) / K.sum(mask)


# def precision(y_true, y_pred):
#     y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
#     upper_tri = tf.linalg.band_part(K.ones_like(y_pred), 0, -1)  # 上三角全1，包括主对角线
#     duijiaoxian = tf.linalg.band_part(K.ones_like(y_pred), 0, 0)  # 主对角线
#     mask = upper_tri - duijiaoxian  # 上三角，不包括对角线
#     return K.sum(y_true*y_pred) / K.sum(y_pred*mask) #分母为0咋办


def recall(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
    return K.sum(y_true*y_pred) / K.sum(y_true)

# Transformer_encoder = build_transformer_model(config_path, checkpoint_path)
# my_zhuanli_layer = Zhuanli(64)
# final_output = my_zhuanli_layer(Transformer_encoder.output)
# model = Model(Transformer_encoder.input, final_output)
# # 调试查看模型里的各层输出和shape
# train_generator = data_generator(train_data, batch_size)
# data_ge = train_generator.__iter__()
# embeding_and_segments, labels = next(data_ge)
# f = K.function(model.input, model.output)
# out = f(embeding_and_segments)
# print(out)
# print(out.shape)
# model.summary()


def model_to_picture(model):
    # 画一个模型结构图
    from keras.utils import plot_model
    plot_model(model, to_file='model_structure.png', show_shapes=True)


def model_summary_tofile(model, file_name="model_summary.txt"):
    from contextlib import redirect_stdout
    with open(file_name, 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=250)


def decode_entities(graph_matric_tensor, n, final_index, R, threshold=0.5, S=None):
    if S is None:
        S = []
    if n == final_index-1:
        R.append(S[:-1])  # 去掉[SEP]
    for i in range(n+1, final_index, 1):  # n+1 到 final_index-1
        if graph_matric_tensor[n, i] >= threshold:
            T = []
            T.extend(S)
            T.append(i)  # 最后会多加一个[SEP]
            decode_entities(graph_matric_tensor, i, final_index, R, S=T)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def __init__(self, model):
        self.model = model

    def recognize(self, text, english_class, threshold=0.5):
        tokens = tokenizer.tokenize(text, maxlen=512)
        chinese_class_to_append = categories_to_chinese[english_class]
        chinese_class_tokens = tokenizer.tokenize(chinese_class_to_append, maxlen=maxlen)[1:-1]
        mapping = tokenizer.rematch(text, tokens)
        tokens.extend(chinese_class_tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        Len = len(token_ids)-len(chinese_class_tokens)
        segment_ids = [0] * Len+[1]*len(chinese_class_tokens)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = self.model.predict([token_ids, segment_ids])[0]
        scores[:, 0] = 0
        entities = []
        result = []
        decode_entities(scores, 0, Len, result, threshold=threshold)

        for e in result:
            entities.append((mapping[e[0]][0], mapping[e[-1]][-1]))
        return entities



def evaluate(data, model):
    """评测函数
    """
    NER = NamedEntityRecognizer(model)
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0], d[1]))
        T = set([tuple(i) for i in d[2:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

# f1, precision, recall = evaluate(valid_data)  # 调试


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model, valid_data):
        super().__init__()
        self.best_val_f1 = 0
        self.model = model
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(self.valid_data, self.model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights('./best_model_cluener_zhuanli.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    model = get_model(False)
    model.load_weights('./best_model_cluener_zhuanli.weights')
    NER = NamedEntityRecognizer(model)
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            l['label'] = {}
            for english, chinese in categories_to_chinese.items():
                entities = NER.recognize(l['text'], english)
                if len(entities) > 0:
                    l['label'][english] = {}
                for start, end in entities:
                    entity = l['text'][start:end+1]
                    if entity not in l['label'][english]:
                        l['label'][english][entity] = []
                    l['label'][english][entity].append([start, end])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


def predict_single_text(text):
    model = get_model(False)
    model.load_weights('./best_model_cluener_zhuanli.weights')
    NER = NamedEntityRecognizer(model)
    res = {}
    for english, chinese in categories_to_chinese.items():
        entities = NER.recognize(text, english)
        if len(entities) > 0:
            res[english] = {}
        for start, end in entities:
            entity = text[start:end+1]
            if entity not in res[english]:
                res[english][entity] = []
            res[english][entity].append([start, end])
    print(res)


def get_model(bert_trainable=False):
    Transformer_encoder = build_transformer_model(config_path, checkpoint_path)
    for layer in Transformer_encoder.layers:
        layer.trainable = bert_trainable
    dropout_output = Dropout(0.2)(Transformer_encoder.output)
    my_zhuanli_layer = Zhuanli(128)
    final_output = my_zhuanli_layer(dropout_output)
    model = Model(Transformer_encoder.input, final_output)
    model.name = "NER"
    # model.summary()
    return model


def train(epochs, train_data, valid_data, bert_trainable=False):
    model = get_model(bert_trainable=bert_trainable)
    if bert_trainable is True:
        # 解冻bert，模型已经训练了后面的分类层
        model.load_weights('./best_model_cluener_zhuanli.weights')
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=[precision, recall]
    )
    evaluator = Evaluator(model, valid_data)
    train_generator = data_generator(train_data, batch_size)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )


if __name__ == '__main__':
    # 标注数据
    train_data = load_data('cluener/train.json')
    valid_data = load_data('cluener/dev.json')
    # test_data = load_data('cluener/test.json')  # 没有label咋办
    categories = list(sorted(categories))
    train(epochs, train_data=train_data, valid_data=valid_data, bert_trainable=False)
    train(fine_tune_epoches, train_data=train_data, valid_data=valid_data, bert_trainable=True)

else:
    model = get_model(False)
    model.load_weights('./best_model_cluener_zhuanli.weights')
    # predict_to_file('/root/ner/cluener/test.json', 'cluener_test.json')
