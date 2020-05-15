import tensorflow as tf
import os

flags = tf.flags
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir", 'data/',
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", 'bert-model/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "ORE", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", 'output/',
    "The output directory where the full model checkpoints will be written."
)

flags.DEFINE_string(
    "pos_output_dir", 'pos_output/',
    "The output directory where the model only use pos features checkpoints will be written."
)

flags.DEFINE_string(
    "dp_output_dir", 'dp_output/',
    "The output directory where the model only use dp features checkpoints will be written."
)

flags.DEFINE_string(
    "only_output_dir", 'only_output/',
    "The output directory where the only BERT model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", 'bert-model/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("pos_embedding_size", 5, "The size of pos_embedding for one feature")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", 'vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")