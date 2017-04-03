from model import Config, Model

import sys, time
import cPickle as pickle
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model_path', None, 'model_path')
flags.DEFINE_string('nbest_path', None, 'nbest_path')
flags.DEFINE_string("vocab_path", None, "vocab_path")
flags.DEFINE_boolean('nbest', False, 'nbest')

FLAGS = flags.FLAGS


def score_all_trees(session, m, nbest):
  """Runs the model on the given data."""
  counts = []
  for step, (trees, original_trees) in enumerate(nbest):
    costs = np.zeros(trees.shape[0])
    state = None
    num = (trees.shape[1] - 1) / 50
    re = (trees.shape[1] - 1) % 50
    if re != 0:
      num += 1
    start = 0
    for i in xrange(num):
      fetches = {"cost": m.cost, "final_state": m.final_state}
      feed_dict = {}
      if i > 0 and state:
        feed_dict[m.initial_state] = state
      shift = re if re > 0 and i == num - 1 else 50
      feed_dict[m.input_data] = trees[:, start:start+shift]
      feed_dict[m.targets] = trees[:,start+1:start+shift+1]
      feed_dict[m.length] = np.ones(trees.shape[0], dtype=np.int) * shift
      
      stuff = session.run(fetches, feed_dict)
      costs += stuff["cost"]
      state = stuff["final_state"]
      start += shift
    am = np.argmin(costs)
    print original_trees[am]["ptb"]


def rerank():
  config = pickle.load(open(FLAGS.model_path + '.config', 'rb'))
  test_nbest_data = reader.read_nbest_trees(FLAGS.vocab_path,
                                            FLAGS.nbest_path)
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = Model(is_training=False, config=config)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      sv.saver.restore(sess, FLAGS.model_path)
      score_all_trees(sess, m, test_nbest_data)

    
def main(_):
  if not FLAGS.vocab_path:
    raise ValueError("Must set --vocab_path to vocab file")
  if not FLAGS.nbest_path:
    raise ValueError("Must set --nbest_path to nbest data")  
  if not FLAGS.model_path:
    raise ValueError("Must set --model_path to model")
  rerank()
    

if __name__ == "__main__":
  tf.app.run()
