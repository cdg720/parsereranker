from random import shuffle
import itertools, sys, time

import cPickle as pickle
import numpy as np
import tensorflow as tf

from model import Config, Model
from utils import chop, run_epoch, evaluate
import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string('data_path', None, 'data_path')
flags.DEFINE_float('init_scale', 0.05, 'init_scale')
flags.DEFINE_float('learning_rate', 0.25, 'learning_rate')
flags.DEFINE_float('max_grad_norm', 20, 'max_grad_norm')
flags.DEFINE_integer('num_layers', 3, 'num_layers')
flags.DEFINE_integer('num_steps', 50, 'num_steps')
flags.DEFINE_integer('hidden_size', 1500, 'hidden_size')
flags.DEFINE_integer('max_epoch', 14, 'max_epoch')
flags.DEFINE_integer('max_max_epoch', 50, 'max_max_epoch')
flags.DEFINE_float('keep_prob', 0.3, 'keep_prob')
flags.DEFINE_float('lr_decay', 0.9, 'lr_decay')
flags.DEFINE_integer('batch_size', 20, 'batch_size')
flags.DEFINE_string('model_path', None, 'model_path')

FLAGS = flags.FLAGS


def train():
  print 'data_path: %s' % FLAGS.data_path
  train_data, valid_data, valid_nbest_data, vocab = \
    reader.ptb_raw_data(FLAGS.data_path)
  train_data = chop(train_data, vocab['<eos>'])
  
  config = Config()
  config.init_scale = FLAGS.init_scale
  config.learning_rate = FLAGS.learning_rate
  config.max_grad_norm = FLAGS.max_grad_norm
  config.num_layers = FLAGS.num_layers
  config.hidden_size = FLAGS.hidden_size
  config.max_epoch = FLAGS.max_epoch
  config.max_max_epoch = FLAGS.max_max_epoch
  config.keep_prob = FLAGS.keep_prob
  config.lr_decay = FLAGS.lr_decay
  config.vocab_size = len(vocab)
  
  print 'init_scale: %.2f' % config.init_scale
  print 'learning_rate: %.2f' % config.learning_rate
  print 'max_grad_norm: %.2f' % config.max_grad_norm
  print 'num_layers: %d' % config.num_layers
  print 'num_steps: %d' % FLAGS.num_steps
  print 'hidden_size: %d' % config.hidden_size
  print 'max_epoch: %d' % config.max_epoch
  print 'max_max_epoch: %d' % config.max_max_epoch
  print 'keep_prob: %.2f' % config.keep_prob
  print 'lr_decay: %.2f' % config.lr_decay
  print 'batch_size: %d' % FLAGS.batch_size
  print 'vocab_size: %d' % config.vocab_size
  sys.stdout.flush()
  
  eval_config = Config()
  eval_config.init_scale = config.init_scale
  eval_config.learning_rate = config.learning_rate
  eval_config.max_grad_norm = config.max_grad_norm
  eval_config.num_layers = config.num_layers
  eval_config.hidden_size = config.hidden_size
  eval_config.max_epoch = config.max_epoch
  eval_config.max_max_epoch = config.max_max_epoch
  eval_config.keep_prob = config.keep_prob
  eval_config.lr_decay = config.lr_decay
  eval_config.vocab_size = len(vocab)

  prev = 0
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = Model(is_training=False, config=eval_config)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      for i in range(config.max_max_epoch):
        shuffle(train_data)
        shuffled_data = list(itertools.chain(*train_data))
      
        start_time = time.time()
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(sess, config.learning_rate * lr_decay)

        print 'Epoch: %d Learning rate: %.3f' % (i + 1, sess.run(m.lr))
        train_perplexity = run_epoch(sess, m, shuffled_data,
                                     FLAGS.batch_size, FLAGS.num_steps,
                                     m.train_op, verbose=True)
        print 'Epoch: %d Train Perplexity: %.3f' % (i + 1, train_perplexity)
        # valid_perplexity = run_epoch(sess, mvalid, valid_data,
        #                              FLAGS.batch_size, FLAGS.num_steps)
        # print 'Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity)
        valid_f1 = evaluate(sess, mvalid, valid_nbest_data)
        print 'Epoch: %d Valid F1: %.2f' % (i + 1, valid_f1)
        print 'It took %.2f seconds' % (time.time() - start_time)
        
        if prev < valid_f1:
          prev = valid_f1
          if FLAGS.model_path:
            print 'Save a model to %s' % FLAGS.model_path
            sv.saver.save(sess, FLAGS.model_path)
            pickle.dump(eval_config, open(FLAGS.model_path + '.config', 'wb'))
        sys.stdout.flush()


def main(_):
  if not FLAGS.data_path:
    raise ValueError('Must set --data_path to PTB data directory')

  print ' '.join(sys.argv)
  train()
    

if __name__ == '__main__':
  tf.app.run()
