import time
import numpy as np


def chop(data, eos):
  new_data = []
  sent = []
  for w in data:
    sent.append(w)
    if w == eos:
      new_data.append(sent)
      sent = []
  return new_data


def ptb_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
    

def run_epoch(session, m, data, batch_size, num_steps,
              eval_op=None, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // batch_size) - 1) // num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = None
  for step, (x, y) in enumerate(ptb_iterator(data, batch_size,
                                             num_steps)):
    fetches = {"cost": m.cost, "final_state": m.final_state}
    if eval_op:
      fetches["op"] = eval_op
    feed_dict = {}
    if state:
      feed_dict[m.initial_state] = state
    feed_dict[m.input_data] = x
    feed_dict[m.targets] = y
    feed_dict[m.length] = np.ones(x.shape[0], dtype=np.int) * x.shape[1]
    stuff = session.run(fetches, feed_dict)
    state = stuff["final_state"]
    costs += np.sum(stuff['cost']) / batch_size
    iters += num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print '%.3f perplexity: %.3f speed: %.0f wps' % \
        (step * 1.0 / epoch_size, np.exp(costs / iters),
         iters * batch_size / (time.time() - start_time))

  return np.exp(costs / iters)


def evaluate(session, m, nbest):
  """Runs the model on the given data."""
  start_time = time.time()
  gold, test, matched = 0, 0, 0
  tree_num = 0
  steps = 50
  for trees, scores, indices in nbest:
    costs = np.zeros(trees.shape[0])
    state = None
    num = (trees.shape[1] - 1) / steps
    re = (trees.shape[1] - 1) % steps
    if re != 0:
      num += 1
    start = 0
    for i in xrange(num):
      fetches = {"cost": m.cost, "final_state": m.final_state}
      feed_dict = {}
      if i > 0 and state:
        feed_dict[m.initial_state] = state
      shift = re if re > 0 and i == num - 1 else steps
      feed_dict[m.input_data] = trees[:, start:start+shift]
      feed_dict[m.targets] = trees[:,start+1:start+shift+1]
      feed_dict[m.length] = np.ones(trees.shape[0], dtype=np.int) * shift
      
      stuff = session.run(fetches, feed_dict)
      costs += stuff["cost"]
      state = stuff["final_state"]
      start += shift
    prev = 0
    for i in indices:
      am = np.argmin(costs[prev:i])
      gold += scores[am+prev][0]
      test += scores[am+prev][1]
      matched += scores[am+prev][2]              
      tree_num += 1
      prev = i
  return 200. * matched / (gold + test)


def unkify(ws):
  uk = 'unk'
  sz = len(ws)-1
  if ws[0].isupper():
    uk = 'c' + uk
  if ws[0].isdigit() and ws[sz].isdigit():
    uk = uk + 'n'
  elif sz <= 2:
    pass
  elif ws[sz-2:sz+1] == 'ing':
    uk = uk + 'ing'
  elif ws[sz-1:sz+1] == 'ed':
    uk = uk + 'ed'
  elif ws[sz-1:sz+1] == 'ly':
    uk = uk + 'ly'
  elif ws[sz] == 's':
    uk = uk + 's'
  elif ws[sz-2:sz+1] == 'est':
    uk = uk + 'est'
  elif ws[sz-1:sz+1] == 'er':
    uk = uk + 'ER'
  elif ws[sz-2:sz+1] == 'ion':
    uk = uk + 'ion'
  elif ws[sz-2:sz+1] == 'ory':
    uk = uk + 'ory'
  elif ws[0:2] == 'un':
    uk = 'un' + uk
  elif ws[sz-1:sz+1] == 'al':
    uk = uk + 'al'
  else:
    for i in xrange(sz):
      if ws[i] == '-':
        uk = uk + '-'
        break
      elif ws[i] == '.':
        uk = uk + '.'
        break
  return '<' + uk + '>'
