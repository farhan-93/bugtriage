import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def clean_str(s):
	"""Clean sentence"""
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	return s.strip().lower()

def load_data_and_labels(filename):
	"""Load sentences and labels"""
	df = pd.read_csv(filename, compression='zip', dtype={'description': object})
	selected = ['assigned_to', 'description']
	#selected = ['assigned_to', 'description', 'summary']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drop non selected columns
	df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
	df = df.reindex(np.random.permutation(df.index))
	#df['merge'] = df.summary + ' ' + df.description
    	# Map the actual labels to one hot labels
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	#x_raw = df['merge'].apply(lambda x: clean_str(x)).tolist()
	x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) - 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word).encode("utf-8"))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors
def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

if __name__ == '__main__':
	input_file = './data/Clean_DS.csv.zip'
	load_data_and_labels(input_file)
