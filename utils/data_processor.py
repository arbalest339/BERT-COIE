import os
import re
import json
import numpy as np
from bert import tokenization
from bert_config import FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos_embedding, dp_embedding, head_embedding, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.pos_embedding = pos_embedding
        self.dp_embedding = dp_embedding
        self.head_embedding = head_embedding


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, pos_embedding, dp_embedding):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.pos_embedding = pos_embedding
        self.dp_embedding = dp_embedding
        # # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):  # 这里是对文件的处理
        """Reads a BIO data."""
        with open(input_file, encoding='utf-8') as f:
            lines = []

            for line in f:
                line = json.loads(line)
                words = ' '.join(list(line['natural']))
                labels = ' '.join(line['tag_seq'])
                poss = line['pos_seq']
                dps = line['dp_seq']
                head = line['head_seq']
                lines.append([labels, words, poss, dps, head])

            return lines


class OreProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.json")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.json")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.json")), "test")

    def get_labels(self):  # 如果改变了这里的标签和训练集的标签，会不会就能进行关系识别
        return ["O", "B-E1", "I-E1", "B-E2", "I-E2", "B-R", "I-R", "X", "[CLS]", "[SEP]"]

    def valid_labels(self):
        return [1, 2, 3, 4, 5, 6]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            pos_embedding = line[2]
            dp_embedding = line[3]
            head_embedding = line[4]
            examples.append(
                InputExample(guid=guid, text=text, label=label, dp_embedding=dp_embedding, head_embedding=head_embedding, pos_embedding=pos_embedding))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, pos_mat, dp_mat, head_mat, label_map, max_seq_length, tokenizer, mode):  # 此时已经确定为最大长度
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
        # print(tokens, labels)           #这里输出保存在record中，basic分词的输出
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        if labels[i] in label_map:
            label_ids.append(label_map[labels[i]])
        else:
            label_ids.append(1)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)



    pos_embedding = np.zeros((max_seq_length, 10), dtype=np.float)
    for i, p in enumerate(example.pos_embedding):
        if i == max_seq_length:  # 达到最大长度，截断
            break
        pos_embedding[i] = pos_mat[p]

    dp_embedding = np.zeros((max_seq_length, 15), dtype=np.float)
    for i, p in enumerate(example.dp_embedding):
        if i == max_seq_length:  # 达到最大长度，截断
            break
        h = example.head_embedding[i]
        if h < max_seq_length:
            dp_embedding[i] = np.concatenate((dp_mat[p], head_mat[h]), axis=-1)
        else:
            pad = np.random.uniform(0, 10, (5,))
            dp_embedding[i] = np.concatenate((dp_mat[p], pad), axis=-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert pos_embedding.shape == (max_seq_length, 10)
    assert dp_embedding.shape == (max_seq_length, 15)
    # # assert len(label_mask) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        pos_embedding=pos_embedding,
        dp_embedding=dp_embedding
        # label_mask = label_mask
    )
    write_tokens(ntokens, mode)
    return feature


def create_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids)
