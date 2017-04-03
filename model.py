import tensorflow as tf

class Config(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 20
  num_layers = 3
  num_steps = 50
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 50
  keep_prob = 0.3
  # correction: for wsj model, we use 0.9. 
  lr_decay = 0.9


class Model(object):
  def __init__(self, is_training, config):
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [None, None])
    self._targets = tf.placeholder(tf.int32, [None, None])
    self._length = tf.placeholder(tf.int32, [None])

    batch_size = tf.shape(self._input_data)[0]
    steps = tf.shape(self._input_data)[1]

    lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    self._initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, self._final_state = tf.nn.dynamic_rnn(
        cell, inputs, sequence_length=self._length,
        initial_state=self._initial_state, dtype=tf.float32)

    output = tf.reshape(outputs, [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    targets = tf.reshape(self._targets, [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    cost = tf.reduce_sum(loss) / tf.to_float(batch_size)
    
    loss = tf.reshape(loss * tf.to_float(tf.sign(targets)),
                      [batch_size, steps])
    self._cost = tf.reduce_sum(loss, 1)

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name='new_learning_rate')
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def input_data(self):
    return self._input_data

  @property
  def length(self):
    return self._length

  @property
  def lr(self):
    return self._lr

  @property
  def targets(self):
    return self._targets

  @property
  def train_op(self):
    return self._train_op
