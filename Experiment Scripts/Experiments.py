import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import AssignProbSent, plot_minimal_pair
from utils import log_uni_unk_prob, sentences_unigram_probability, calculating_perplexity
import utils
from sentence_manipulation import find_sentence_that_have, comparative_sent_printing, x2y
count_sign = '__COUNT__'


modelname = 'TrieRoot'
# Trained on Harry Potter 1-7
infile = open(modelname, 'rb')
root = pickle.load(infile)
infile.close()

filename = 'eval_dataset'
infile = open(filename, 'rb')
eval_set = pickle.load(infile)
infile.close()

filename='is_sentences'
infile = open(filename,'rb')
is_sentences=pickle.load(infile)
infile.close()


bad_is_sentences = x2y(is_sentences, 'is','are')
unk_is_sentences = x2y(is_sentences, 'is','who')
a = is_sentences[6][1:9]
b = bad_is_sentences[6][1:9]
plot_minimal_pair(a, b, root, AssignProbSent, __plt__=plt)