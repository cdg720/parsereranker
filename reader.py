import os

import numpy as np

from utils import unkify
import collections, gzip


def build_vocab(filename):
  data = read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  words = ["<pad>"] + list(words)
  word2id = dict(zip(words, range(len(words))))

  return word2id


def read_words(filename):
  with open_file(filename) as f:
    return f.read().replace('\n', '<eos>').split()


def trees_file2word_ids(filename, word2id):
  data = read_words(filename)
  return [word2id[word] for word in data]


def open_file(path):
  if path.endswith('.gz'):
    return gzip.open(path, 'rb')
  else:
    return open(path, 'r')


# read preprocessed nbest
def nbest_file2word_ids(filename, word2id):
  data = []
  scores = []
  trees = []
  count = 0
  with open_file(filename) as f:
    for line in f:
      if count == 0:
        count = int(line)
      elif not line.startswith(' '):
        tmp = line.split()
        gold = int(tmp[0])
        test = int(tmp[1])
        matched = int(tmp[2])
        scores.append((gold, test, matched))
      else:
        line = line.replace('\n', '<eos>').split()
        line = [word2id["<eos>"]] + [word2id[word] for word in line]
        trees.append(line)
        count -= 1
        if count == 0:
          max_len = max([len(t) for t in trees])
          for i in xrange(len(trees)):
            t_len = len(trees[i])
            for j in xrange(max_len - t_len):
              trees[i].append(word2id["<pad>"])
          data.append((np.array(trees), scores))
          trees = []
          scores = []
  data.sort(key = lambda x: x[0].shape[1])
  batched_data = []
  batched_trees = []
  batched_scores = []
  indices = []
  for trees, scores in data:
    batched_trees.append(trees)
    batched_scores.extend(scores)
    num = len(batched_scores)
    indices.append(num)
    if num > 500:
      max_len = batched_trees[-1].shape[1]
      for i in xrange(len(batched_trees) - 1):
        this_len = batched_trees[i].shape[1]
        if max_len > this_len:
          batched_trees[i] = np.lib.pad(
              batched_trees[i], ((0, 0), (0, max_len - this_len)),
              'constant', constant_values=0)
      batched_data.append((np.concatenate(batched_trees, axis=0),
                          batched_scores, indices))
      batched_trees = []
      batched_scores = []
      indices = []
  if batched_trees:
    max_len = batched_trees[-1].shape[1]
    for i in xrange(len(batched_trees) - 1):
      this_len = batched_trees[i].shape[1]
      if max_len > this_len:
        batched_trees[i] = np.lib.pad(
            batched_trees[i], ((0, 0), (0, max_len - this_len)),
            'constant', constant_values=0)
    batched_data.append((np.concatenate(batched_trees, axis=0),
                         batched_scores, indices))
  return batched_data


def _generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0:
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def _process_tree(line, words, tags=False):
  tokens = line.replace(')', ' )').split()
  nonterminals = []
  new_tokens = []
  pop = False
  ind = 0
  for token in tokens:
    if token.startswith('('): # open paren
      new_token = token[1:]
      nonterminals.append(new_token)
      new_tokens.append(token)
    elif token == ')': # close paren
      if pop: # preterminal
        pop = False
      else: # nonterminal
        new_token = ')' + nonterminals.pop()
        new_tokens.append(new_token)
    else: # word
      if not tags:
        tag = '(' + nonterminals.pop() # pop preterminal
        new_tokens.pop()
        pop = True
      if token.lower() in words:
        new_tokens.append(token.lower())
      else:
        new_tokens.append(unkify(token))
  return ' ' + ' '.join(new_tokens[1:-1]) + ' '


def _remove_duplicates(nbest):
  new_nbest = []
  seqs = set()
  for t in nbest:
    if t['seq'] not in seqs:
      seqs.add(t['seq'])
      new_nbest.append(t)
  return new_nbest


# read silver data
def silver_file2word_ids(filename):
  for line in open_file(filename):
    yield [int(x) for x in line.split()] 


# read data for training.
def ptb_raw_data(data_path=None, with_silver=False):
  train_path = os.path.join(data_path, "train.gz")
  valid_path = os.path.join(data_path, "dev.gz")
  valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word2id = build_vocab(train_path)
  train_data = trees_file2word_ids(train_path, word2id)
  valid_data = trees_file2word_ids(valid_path, word2id)
  valid_nbest_data = nbest_file2word_ids(valid_nbest_path, word2id)
  if with_silver:
    silver_path = os.path.join(data_path, 'silver.gz')
    return train_data, silver_path, valid_data, valid_nbest_data, word2id
  else:
    return train_data, valid_data, valid_nbest_data, word2id


def read_nbest_trees(vocab_path, nbest_path):
  word2id = {}
  for line in open_file(vocab_path):
    word, word_id = line.split()
    word2id[word] = int(word_id)
  nbest_data = file2nbest_trees(nbest_path, word2id)
  return nbest_data


def file2nbest_trees(filename, word2id):
  data = []
  for ts in _generate_nbest(open_file(filename)):
    for t in ts:
      t['seq'] = _process_tree(t['ptb'], word2id)
    ts = _remove_duplicates(ts)
    trees = []
    for t in ts:
      trees.append([word2id["<eos>"]] + [word2id[word] for word in \
                    t['seq'].split()] + [word2id['<eos>']])
    max_len = max([len(t) for t in trees])
    for i in xrange(len(trees)):
      t_len = len(trees[i])
      for j in xrange(max_len - t_len):
        trees[i].append(word2id["<pad>"])
    data.append((np.array(trees), ts))
  return data
