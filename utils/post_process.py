import os
import json
import argparse


def fix_incomplete_triplets(predlines, dplines, headlines):
    for i, predline in enumerate(predlines):
        # 对R的补足
        if not "B-R" in predline:
            if "B-E1" in predline:
                eindex = predline.index("B-E1")
                head = headlines[i][eindex]
                if head < len(predline):
                    predlines[i][head] = 'B-R'
                    for j in range(head + 1, ):
                        if dplines[i][j] == 13:  # HED
                            predlines[i][head] = 'I-R'
                        else:
                            break
            if "B-E2" in predline:
                eindex = predline.index("B-E2")
                head = headlines[i][eindex]
                if head < len(predline):
                    predlines[i][head] = 'B-R'
                    for j in range(head + 1, ):
                        if dplines[i][j] == 13:  # HED
                            predlines[i][head] = 'I-R'
                        else:
                            break
        else:  # 对E的补足
            rindex = predline.index("B-R")
            if not "B-E1" in predline:
                flag = False
                for j, w in enumerate(predline):
                    if dplines[i][j] == 0 and headlines[i][j] == rindex:
                        predlines[i][j] = 'B-E1' if not flag else 'I-E1'
            if not "B-E2" in predline:
                flag = False
                for j, w in enumerate(predline):
                    if dplines[i][j] in [1, 2, 3] and headlines[i][j] == rindex:
                        predlines[i][j] = 'B-E2' if not flag else 'I-E2'
    return predlines


def emerge(data_dir, output_dir):
    nerpath = os.path.join(data_dir, "test.json")
    orepath = os.path.join(output_dir, "label_test.txt")
    outpath = os.path.join(output_dir, "emerge.txt")
    rerrorpath = os.path.join(output_dir, "relation_error.txt")
    eerrorpath = os.path.join(output_dir, "argument_error.txt")

    nerfile = open(nerpath, 'r', encoding='utf-8')
    orefile = open(orepath, 'r', encoding='utf-8')
    emerge = open(outpath, 'w')
    rerror = open(rerrorpath, 'w')
    eerror = open(eerrorpath, 'w')

    labels_map = {"O": 0, "B-E1": 1, "I-E1": 2, "B-E2": 3, "I-E2": 4, "B-R": 5, "I-R": 6, "X": 7}
    ent_labels = ["B-E1", "I-E1", "B-E2", "I-E2", "B-R", "I-R"]
    invalid_labels = ["O", "X"]
    begin_ids = ["B-E1", "B-E2", "B-R"]

    correct = 0
    redundancy_correct = 0
    miss_correct = 0
    gold_entities_num = 0
    pred_entities_num = 0
    r_num = 0
    e_num = 0
    r_pred = 0
    e_pred = 0
    r_correct = 0
    e_correct = 0
    lines = nerfile.readlines()
    goldlines = [json.loads(goldline)['tag_seq'] for goldline in lines]
    predlines = [predline.strip().split(' ') for predline in orefile.readlines()]
    headlines = [json.loads(goldline)['head_seq'] for goldline in lines]
    dplines = [json.loads(goldline)['dp_seq'] for goldline in lines]
    toklines = [json.loads(line)['natural'] for line in lines]
    logiclines = [json.loads(line)['logic'] for line in lines]
    poslines = [json.loads(line)['pos_seq'] for line in lines]
    predlines.pop(-1)
    for i, predline in enumerate(predlines):
        if "[CLS]" in predline:
            predlines[i].remove("[CLS]")
            predlines[i].remove("[SEP]")

    for i, gold in enumerate(goldlines):
        pred = predlines[i]
        for j in range(len(gold)):
            if gold[j] in begin_ids:
                gold_entities_num += 1

        for j in range(len(pred)):
            if pred[j] in begin_ids:
                pred_entities_num += 1

        predlines = fix_incomplete_triplets(predlines, dplines, headlines)

        pred_entities_pos = []
        gold_entities_pos = []
        start, end = 0, 0

        for j in range(len(gold)):
            if gold[j] in begin_ids:
                start = j
                for k in range(j + 1, len(gold)):

                    if gold[k] in ent_labels:
                        continue

                    if gold[k] in invalid_labels or gold[k] in begin_ids:
                        end = k - 1
                        break
                else:
                    end = len(gold) - 1
                if 'E' in gold[j]:
                    e_num += 1
                    gold_entities_pos.append((start, end, 'E'))
                else:
                    r_num += 1
                    gold_entities_pos.append((start, end, 'R'))

        for j in range(len(pred)):
            if pred[j] in begin_ids:
                start = j
                for k in range(j + 1, len(pred)):

                    if gold[k] in ent_labels:
                        continue

                    if pred[k] in invalid_labels or pred[k] in begin_ids:
                        end = k - 1
                        break
                else:
                    end = len(pred) - 1

                if 'E' in pred[j]:
                    e_pred += 1
                    pred_entities_pos.append((start, end, 'E'))
                else:
                    r_pred += 1
                    pred_entities_pos.append((start, end, 'R'))

        for k, miss in enumerate(pred_entities_pos):
            start, end, _ = miss
            new_start = start
            new_end = end
            for j in range(start, 0, -1):
                if dplines[i][j] == dplines[i][start] and poslines[i][j] == poslines[i][start]:
                    new_start = j
                    predlines[i][j] = predlines[i][start].replace('I', 'B')
            for j in range(end, -1, -1):
                if dplines[i][j] == dplines[i][end] and poslines[i][j] == poslines[i][end]:
                    new_end = j
                    predlines[i][j] = predlines[i][start].replace('B', 'I')
            pred_entities_pos[k] = (new_start, new_end, _)

        miss_corrects = []
        for entity in pred_entities_pos:
            if entity in gold_entities_pos:
                correct += 1
                if entity[2] == 'E':
                    e_correct += 1
                else:
                    r_correct += 1
            else:
                for gold in gold_entities_pos:
                    if entity[0] >= gold[0] and entity[1] <= gold[1] and entity[2] == gold[2]:
                        redundancy_correct += 1
                    elif entity[0] <= gold[0] and entity[1] >= gold[1] and entity[2] == gold[2]:
                        miss_correct += 1
                        miss_corrects.append(entity)
                if entity[2] == 'R':
                    extractions = toklines[i][entity[0]:entity[1] + 1]
                    rerror.write(''.join(toklines[i]) + '\n')
                    rerror.write(str(logiclines[i]) + '\n')
                    rerror.write(''.join(extractions) + '\n')
                else:
                    extractions = toklines[i][entity[0]:entity[1] + 1]
                    eerror.write(''.join(toklines[i]) + '\n')
                    eerror.write(str(logiclines[i]) + '\n')
                    eerror.write(''.join(extractions) + '\n')

    for i, predline in enumerate(predlines):
        extractions = []
        for j, l in enumerate(predlines[i]):
            if 'B' in l:
                extractions.append(',' + toklines[i][j])
            elif not l == 'O':
                extractions.append(toklines[i][j])

        emerge.write(''.join(toklines[i]) + '\n')
        emerge.write(str(logiclines[i]) + '\n')
        emerge.write(''.join(extractions) + '\n')
        emerge.flush()

    print("Report precision, recall, and f1:")
    acc_p = correct / pred_entities_num
    acc_r = correct / gold_entities_num
    acc_f1 = 2 * acc_p * acc_r / (acc_p + acc_r)
    fuzzy_p = (correct + miss_correct) / pred_entities_num
    fuzzy_r = (correct + miss_correct) / gold_entities_num
    fuzzy_f1 = 2 * fuzzy_p * fuzzy_r / (fuzzy_p + fuzzy_r)
    print("accurate: {:.3f}, {:.3f}, {:.3f}; fuzzy: {:.3f}, {:.3f}, {:.3f}".format(acc_p, acc_r, acc_f1, fuzzy_p,
                                                                                   fuzzy_r, fuzzy_f1))
    print("argument_correct {},  relation_correct {}".format(e_correct, r_correct))
    print("argument_num {},  relation_num {}".format(e_num, r_num))
    print("argument_pred {},  relation_pred {}".format(e_pred, r_pred))


def _argparse():
    parser = argparse.ArgumentParser(description="Add POS and DP features into the data set")
    parser.add_argument('-data_dir', action='store', dest='data_dir', required=True,
                        default="saoke_filtered.json", help='The path of testing set')
    parser.add_argument('-output_dir', action='store', dest='output_dir', required=True,
                        default="saoke_filtered.json", help='The path of model output dir')
    return parser.parse_args()


if __name__ == "__main__":
    arg = _argparse()
    emerge(arg.data_dir, arg.output_dir)
