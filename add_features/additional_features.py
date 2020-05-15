import os
import json
import re
import argparse
from srl_labler import SRL_Labler

posmap = {"a": 0,
          "b": 1,
          "c": 2,
          "d": 3,
          "e": 4,
          "g": 5,
          "h": 6,
          "i": 7,
          "j": 8,
          "k": 9,
          "m": 10,
          "n": 11,
          "nd": 12,
          "nh": 13,
          "ni": 14,
          "nl": 15,
          "ns": 16,
          "nt": 17,
          "nz": 18,
          "o": 19,
          "p": 20,
          "q": 21,
          "r": 22,
          "u": 23,
          "v": 24,
          "wp": 25,
          "ws": 26,
          "x": 27,
          "z": 28,
          "%": 29}
dpmap = {"SBV": 0,
         "VOB": 1,
         "IOB": 2,
         "FOB": 3,
         "DBL": 4,
         "ATT": 5,
         "ADV": 6,
         "CMP": 7,
         "COO": 8,
         "POB": 9,
         "LAD": 10,
         "RAD": 11,
         "IS": 12,
         "HED": 13,
         "WP": 14}


def gettag(words, postags, arcs):
    poss = []
    dps = []
    head = []
    headmap = {}
    i = 0
    for fetch in zip(words, postags, arcs):
        word, pos, arc = fetch
        for w in word:
            poss.append(posmap[pos])
            dps.append(dpmap[arc.relation])
            headmap[arc.head] = i
    for fetch in zip(words, postags, arcs):
        word, pos, arc = fetch
        for w in word:
            head.append(headmap[arc.head])

    return poss, dps, head


def _argparse():
    parser = argparse.ArgumentParser(description="Add POS and DP features into the data set")
    parser.add_argument('-sp', action='store_true', dest='sp',
                        default=False, help='Whether to split the data set to training and testing set')
    parser.add_argument('-ltp',  action='store', dest='ltp_data_dir', required=True,
                        default="saoke_filtered.json", help='The path of source data set')
    parser.add_argument('-rf', action='store', dest='rf',
                        default="saoke_filtered.json", help='The path of source data set')
    parser.add_argument('-wf', action='store', dest='wf',
                        default="saoke_out.json", help='The path of output data set, use when do not split')
    parser.add_argument('-train', action='store', dest='train',
                        default="train.json", help='The path of train data set path')
    parser.add_argument('-test', action='store', dest='test',
                        default="test.json", help='The path of test data set path')
    return parser.parse_args()


if __name__ == "__main__":
    arg = _argparse()

    rf = open(arg.rf, 'r', encoding='utf-8')
    if arg.sp:
        wf1 = open(arg.train, 'w', encoding='utf-8')
        wf2 = open(arg.test, 'w', encoding='utf-8')
    else:
        wf = open(arg.wf, 'w', encoding='utf-8')
    labeler = SRL_Labler(arg.ltp_path_dir)
    lines = rf.readlines()
    line_num = len(lines)

    for i, line in enumerate(lines):
        sentence = json.loads(line)
        natural = re.sub(' ', '', sentence['natural'])
        words, postags, arcs = labeler.get_features(natural)
        poss, dp, head = gettag(words, postags, arcs)
        sentence['natural'] = natural
        sentence['pos_seq'] = poss
        sentence['dp_seq'] = dp
        sentence['head_seq'] = head
        s = json.dumps(sentence, ensure_ascii=False) + '\n'
        if arg.sp:
            if i < line_num * 9 // 10:
                wf1.write(s)
            else:
                wf2.write(s)
        else:
            wf.write(s)
