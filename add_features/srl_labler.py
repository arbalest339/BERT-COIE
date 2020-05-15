# -*- coding: utf-8 -*-
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
from pyltp import SementicRoleLabeller


class SRL_Labler(object):
    def __init__(self, ltp_data_dir):
        # ltp模型目录的路径
        self.cws_model_path = os.path.join(ltp_data_dir, 'cws.model')  # 分词模型
        self.pos_model_path = os.path.join(ltp_data_dir, 'pos.model')  # 词性标注模型
        self.par_model_path = os.path.join(ltp_data_dir, 'parser.model')  # 依存分析模型

        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(self.cws_model_path)  # 加载模型

        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

        self.parser = Parser()  # 初始化实例
        self.parser.load(self.par_model_path)  # 加载模型

    def __del__(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()  # 释放模型

    def get_features(self, sentence):
        words = self.segmentor.segment(sentence)  # 分词
        postags = self.postagger.postag(words)  # 词性标注
        arcs = self.parser.parse(words, postags)  # 句法分析

        words = list(words)
        postags = list(postags)
        arcs = list(arcs)

        # 打印结果
        # for role in roles:
        #     print(role.index, " ".join(
        #         ["%s:(%s)" % (arg.name, words[arg.range.start:arg.range.end]) for arg in role.arguments]))

        return words, postags, arcs
