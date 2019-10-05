"""Generate language using XLNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import re

from tqdm import tqdm
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
# tf.enable_eager_execution()
import sentencepiece as spm
import collections

from prepro_utils import preprocess_text, encode_ids
import model
import beam_search

special_symbols = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<pad>"  : 3,
    "<eod>"  : 4,
    "<eop>"  : 5,
    "<hi>"   : 6,
    "<eng>"   : 7
}

SOS_ID = special_symbols['<s>']
EOS_ID = special_symbols['</s>']
UNK_ID = special_symbols["<unk>"]
EOD_ID = special_symbols["<eod>"]
EOP_ID = special_symbols["<eop>"]
HIN_ID = special_symbols["<hi>"]
ENG_ID = special_symbols["<eng>"]
PAD_ID = special_symbols["<pad>"]

parser = argparse.ArgumentParser()

# TPU parameters
parser.add_argument("--master", default=None,
                    help="master", type=str)
parser.add_argument("--tpu", default=None,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.",
      type=str)
parser.add_argument("--gcp_project", default=None,
      help="Project name for the Cloud TPU-enabled project. If not specified, "
      "we will attempt to automatically detect the GCE project from metadata.",
      type=str)
parser.add_argument("--tpu_zone",default=None,
      help="GCE zone where the Cloud TPU is located in. If not specified, we "
      "will attempt to automatically detect the GCE project from metadata.",
      type=str)
parser.add_argument("--use_tpu", action='store_true',
      help="Use TPUs rather than plain CPUs.")
parser.add_argument("--num_hosts", default=1,
      help="number of TPU hosts", type=int)
parser.add_argument("--num_core_per_host", default=8,
      help="number of cores per host",type=int)
parser.add_argument("--iterations", default=500,
      help="Number of iterations per repeat loop for TPU.")


# Model
parser.add_argument("--n_layer", default=6, type=int,
      help="Number of layers.")
parser.add_argument("--d_model", default=500, type=int,
      help="Dimension of the model.")
parser.add_argument("--d_embed", default=500, type=int,
      help="Dimension of the embeddings.")
parser.add_argument("--n_head", default=10, type=int,
      help="Number of attention heads.")
parser.add_argument("--d_head", default=50, type=int,
      help="Dimension of each attention head.")
parser.add_argument("--d_inner", default=1000, type=int,
      help="Dimension of inner hidden size in positionwise feed-forward.")
parser.add_argument("--untie_r", action='store_true',
      help="untie r_w_bias and r_r_bias")
parser.add_argument("--clamp_len", default=-1,
                    help="Clamp length", type=int)
parser.add_argument("--same_length", action='store_true', default=False,
                    help="Same length attention")
parser.add_argument("--tie_weight", type=bool, default=True,
      help="Tie embedding and softmax weight.")
# Data and memory
parser.add_argument("--seq_len", default=70,
      help="Maxmium number of steps in the input", type=int)
parser.add_argument("--n_token", default=32000, help='vocab size', type=int)
parser.add_argument("--batch_size", default=1, help='batch size', type=int)
parser.add_argument("--uncased", default=False, action='store_true',
                    help="Use uncased inputs or not.")

# I/O paths
parser.add_argument("--init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
parser.add_argument("--spiece_model_file", default="",
                    help="Sentence Piece model path.")
parser.add_argument("--input_file", default="",
                    help="File containing prompts separated by empty new line "
                    "for conditional sampling")
parser.add_argument("--output_dir",default='prediction',
                    help="output dir for saving predicted results"
                    "if input file is provided. Doesn't apply to interactive",
                    type=str)

# prediction
parser.add_argument(
    "--interactive",
    help="Flag for interactive prediction through command line",
    action='store_true')
parser.add_argument("--beam_size",default=4,type=int,
    help="Beam width for beam search decoding")
parser.add_argument("--beam_alpha",default=0.6,type=float,
    help="alpha parameter for beam search decoding")
parser.add_argument("--max_decode_length", default=1024,
                    help="Maximum Number of tokens to predict", type=int)

# NMT specifics
parser.add_argument("--bi_mask",action="store_true",
                     help="Use bidirectional mask for source tokens")
parser.add_argument("--use_sos", default=False, action='store_true',
                    help="whether to use SOS.")
parser.add_argument("--transliterate", action="store_true",
                  help="Transliterate to hindi.")
parser.add_argument("--src_lang", default='english',
                    help="Source lang english/hindi.")
parser.add_argument("--tgt_lang", default='hindi',
                    help="Target lang english/hindi.")

FLAGS = parser.parse_args()
# Warning: global variable
sp = spm.SentencePieceProcessor()
sp.Load(FLAGS.spiece_model_file)


def get_preprocessor(examples, tokenize_fn):
    """
    Input:
    examples: [List[str]] input texts
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    def generator():
        for i in range(0,len(examples),FLAGS.batch_size):
            batched = examples[i:i+FLAGS.batch_size]
            tokens_batched = list(map(tokenize_fn,batched))
            maxlen = max(map(len,tokens_batched))
            for tokens in tokens_batched:
                pad_len = maxlen-len(tokens)
                src_id = [SOS_ID]
                ids = src_id + tokens + [EOS_ID]
                
                masks = [0]*pad_len+[1]*len(ids)
                ids = [PAD_ID]*pad_len+ids

                ids = ids[-FLAGS.seq_len:]
                masks = masks[-FLAGS.seq_len:]

                yield {'input':ids,'input_mask':masks}

    return generator


def get_input_dataset(preprocessor,batch_size):
    """Returns tf.data.Dataset for input"""

    dataset = tf.data.Dataset.from_generator(preprocessor,
                                             output_types={'input':tf.int32,
                                             'input_mask':tf.float32})
    dataset = dataset.batch(batch_size,
                            drop_remainder=False)
    dataset.prefetch(1)
    return dataset

def get_tpu_input_dataset(examples,tokenize_fn,batch_size):
    
    # remainder = len(examples)%batch_size
    # examples = examples + [examples[-1]]*remainder
    maxm = FLAGS.seq_len
    ids = list(map(tokenize_fn,examples))
    ids = [[SOS_ID]+i+[EOS_ID] for i in ids]

    masks = [[0]*max(0,maxm-len(d))+[1]*len(d) for d in ids]
    ids = [[PAD_ID]*max(0,maxm-len(d))+d for d in ids]
    
    ids_tensors = [tf.constant(d[:maxm],dtype=tf.int32) for d in ids]
    masks_tensors = [tf.constant(m[:maxm],dtype=tf.float32) for m in masks]
    dataset = tf.data.Dataset.from_tensor_slices({'input':ids,
                                            'input_mask':masks_tensors})
    dataset = dataset.batch(batch_size)
    dataset.prefetch(1)
    return dataset

def get_logits(input_ids,mems,input_mask,target_mask):
    """Builds the graph for calculating the final logits"""
    is_training = False

    cutoffs = []
    train_bin_sizes = []
    eval_bin_sizes = []
    proj_share_all_but_first = True
    n_token = FLAGS.n_token

    batch_size = FLAGS.batch_size

    features = {"input": input_ids}
    inp = tf.transpose(features["input"], [1, 0])
    input_mask = tf.transpose(input_mask, [1, 0])
    target_mask = tf.transpose(target_mask, [1, 0])
    tgt = None

    inp_perms, tgt_perms, head_tgt = None, None, None

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    tf.logging.info("Vocab size : {}".format(n_token))
    tf.logging.info("Batch size : {}".format(batch_size))

    logits, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=0,
        dropatt=0,
        initializer=initializer,
        is_training=is_training,
        mem_len=FLAGS.seq_len+FLAGS.max_decode_length,
        cutoffs=cutoffs,
        div_val=1,
        tie_projs=tie_projs,
        input_perms=inp_perms,
        target_perms=tgt_perms,
        head_target=head_tgt,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=FLAGS.use_tpu,
        untie_r=FLAGS.untie_r,
        proj_same_dim=True,
        bidirectional_mask=FLAGS.bi_mask,
        infer=True,
        target_mask=target_mask,
        input_mask=input_mask,
        tgt_len=1)

    return logits,new_mems


def cache_fn(batch_size):
    if not FLAGS.use_tpu:
      return None
    mem_len = FLAGS.seq_len+FLAGS.max_decode_length
    mems = []
    for l in range(FLAGS.n_layer):
      if mem_len > 0:
        mems.append(
          tf.zeros([mem_len, batch_size, FLAGS.d_model], dtype=tf.float32))
      else:
        mems.append(tf.zeros([mem_len], dtype=tf.float32))
    # for mem_mask
    if mem_len > 0:
      mems.append(tf.zeros([mem_len, batch_size], dtype=tf.float32))
    else:
      mems.append(tf.zeros([mem_len], dtype=tf.float32))

    return mems


def model_fn(features,labels,mode,params):
    """Gets features and
    return predicted tokens)
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """

    assert mode==tf.estimator.ModeKeys.PREDICT, "Only PREDICT mode supported"
    batch_size = tf.shape(features['input'])[0]
    input_tensor = features['input']

    # Calculating hidden states of inputs and getting latest logit
    input_mask = features['input_mask']
    target_mask = tf.ones((tf.shape(input_tensor)[0],1))
    mems = cache_fn(tf.shape(input_mask)[0])
    _,mems = get_logits(input_tensor,mems=mems,input_mask=input_mask,
                             target_mask=target_mask)

        
    mems = {i:tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
                  tf.transpose(mems[i],[1,0])
                  for i in range(len(mems))}
    # logits = tf.reshape(logits,(batch_size,1,-1))
    # latest_toks,latest_confs = sample_token(logits) 
    # all_confs = latest_confs
    # all_toks = latest_toks


    def symbols_to_logits_fn(toks,_,mems):
        # We need only last token
        toks = toks[:,-1:]
        # input_mask set all the inputs to be valid
        input_mask = tf.ones_like(toks,dtype=tf.float32)
        # target_mask set to be of ones
        target_mask = tf.ones((tf.shape(toks)[0],1),dtype=tf.float32)
        mems = [tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
                tf.transpose(mems[i],[1,0])
                for i in range(len(mems))]
        logits,mems = get_logits(toks,mems=mems,input_mask=input_mask,
                                         target_mask=target_mask)
        return logits,{i:tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
                      tf.transpose(mems[i],[1,0])
                      for i in range(len(mems))}
    
    initial_ids = tf.ones((batch_size),dtype=tf.int32)*SOS_ID


    decoded_ids, scores = beam_search.sequence_beam_search(
    symbols_to_logits_fn, initial_ids, mems, FLAGS.n_token, FLAGS.beam_size,
    FLAGS.beam_alpha, FLAGS.max_decode_length, EOS_ID, padded_decode=FLAGS.use_tpu)
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    scaffold_fn = init_from_checkpoint_scaffold()
    spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={
          "top_ids":top_decoded_ids,
          "top_scores":top_scores
          },
        scaffold_fn=scaffold_fn
        )


    return spec


             

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    # tf.logging.info('original name: %s', name)
    if name not in name_to_variable:
      continue
    # assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def init_from_checkpoint_scaffold(global_vars=False):
  tvars = tf.global_variables() if global_vars else tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if FLAGS.init_checkpoint is not None:
    if FLAGS.init_checkpoint.endswith("latest"):
      ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
      init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    else:
      init_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if FLAGS.use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
  return scaffold_fn

def transliterate_back(text,lang):
  # English return as it is
  if text=="":
    return text
  if lang==0:
    return text
  from cltk.corpus.sanskrit.itrans.unicode_transliterate import ItransTransliterator as its
  return its.from_itrans(text,'hi')

def get_input_fn(lines):

  def tokenize_fn(text):
      text = preprocess_text(text, lower=FLAGS.uncased)
      text = encode_ids(sp, text,
                         transliterate=FLAGS.transliterate, language_tag=True,
                         eng_id=ENG_ID, hin_id=HIN_ID)
      return text

  def input_fn(params):
    preprocessor = get_preprocessor(lines,tokenize_fn)
    dataset = get_input_dataset(preprocessor,batch_size=params['batch_size'])
    return dataset

  def tpu_input_fn(params):
    return get_tpu_input_dataset(lines,tokenize_fn,params['batch_size'])

  if FLAGS.use_tpu:
    return tpu_input_fn

  return input_fn

def main():
    """Main function routine"""

    tf.logging.set_verbosity(tf.logging.INFO)    


    to_special_symbol = {v:k for k,v in special_symbols.items()}
    def parse_ids(toks):
        """Uses sentencepiece to conver to text. Subsitute
        EOP_ID and EOD_ID with new lines, and rest with their names"""
        
        # IF EOS_ID was encountered rest will be pad ids
        print(toks)
        if EOS_ID in toks:
            toks = toks[:toks.index(EOS_ID)]

        sent = sp.decode_ids(toks)
        if FLAGS.transliterate and FLAGS.tgt_lang!='english':
          sent = transliterate_back(sent,FLAGS.tgt_lang)


        return sent

    gpu_options = tf.GPUOptions(allow_growth=True)

    session_config=tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=gpu_options,
                                  log_device_placement=True)

    # TPU Configuration
    if FLAGS.use_tpu:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      tpu_cluster_resolver = None
    per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      session_config=session_config,
      tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations,
      num_shards=FLAGS.num_core_per_host * FLAGS.num_hosts,
      per_host_input_for_training=per_host_input),
     )
    # TPU Estimator
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size)

    def predict(examples):
        """Given a list of texts in examples
        return the result"""
        input_fn = get_input_fn(examples)
        result = estimator.predict(input_fn=input_fn)
        i=0
        for res in result:
          yield res['top_ids'],res['top_scores']
          i+=1
          if i==len(examples):
            # TPU uses more examples for filling the batch
            break

    if FLAGS.interactive:
        tf.logging.info("Interactive flag received."
                        " Ignoring input files if any.")
        while True:
            text = input("----PROMPT----\n")
            outputs = predict([text])
            print(outputs)
            output = next(outputs)
            out = parse_ids(output[0].tolist())
            print("======Translation======")
            print(out)
            print("=====================")
    else:
        assert FLAGS.input_file!="", "Please provide either an"\
        " input file or set interactive flag for command line input"
        assert os.path.exists(FLAGS.input_file), FLAGS.input_file+\
        " does not exists"

        with open(FLAGS.input_file) as f:
            texts = []
            for line in f:
                texts.append(line.strip())

        tf.logging.info("Got %s lines in the input file",
                        len(texts))
        outputs = predict(texts)
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        with open(os.path.join(FLAGS.output_dir,FLAGS.input_file),'w') as f:
            for i in range(0,len(texts)):
                output,_ = next(outputs)
                out = parse_ids(output.tolist())
                f.write(out+'\n')
# Fixed flags
FLAGS.dropout = 0
FLAGS.dropatt = 0
FLAGS.init = "normal"
FLAGS.init_std = 0.02
FLAGS.init_range = 0.1
FLAGS.proj_init_std = 0.01

if __name__ == "__main__":

    
    print("Args: {}".format(vars(FLAGS)))
    main()
