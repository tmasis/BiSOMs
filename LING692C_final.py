import numpy as np
from pyERA.hebbian import HebbianNetwork
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
import nltk
from nltk import bigrams
import itertools
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import epitran
import os
import random
import re


# Makes co-occurrence matrix
def co_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab) 
    vocab_i = {word: i for i, word in enumerate(vocab)}
    bi_grams = list(bigrams(corpus))
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    comat = np.zeros((len(vocab),len(vocab)))
    for b in bigram_freq:
        current = b[0][1]
        prev = b[0][0]
        count = b[1]
        pos_current = vocab_i[current]
        pos_prev = vocab_i[prev]
        comat[pos_current][pos_prev] = count
    comat = np.matrix(comat)
    return comat, vocab_i


## Process data
print("Collecting data...")
orig_folders = [#"/mnt/c/Users/Tessa/Documents/umass/LING692c_final/engcan_transcripts/Can/TimCan/",
          #'/mnt/c/Users/Tessa/Documents/umass/LING692c_final/engcan_transcripts/Eng/TimEng/']#,
          '/mnt/c/Users/Tessa/Documents/umass/LING692c_final/engcan_transcripts/Eng/TimMixed/']

# Data is list of lists where each list is a sentence and each element is a word
data = []
# For each folder
for folder in orig_folders:
    # For each file
    for f in os.listdir(folder):
        if f.endswith('.txt'):
            openf = open(folder+f, 'r')
            # For line in file
            for line in openf: data.append(word_tokenize(line))


## phonRep is a matrix where each vector is a phonetic representation
## semRep is a matrix where each vector is a word embedding
create = False

# Generate phonemic representations
# Convert each word to unicode and then to IPA
print("Generating phonemic representations...")
if create:
    phonData= []
    eng = epitran.Epitran('eng-Latn')
    chi = epitran.Epitran('cmn-Hant', cedict_file='cedict_1_0_ts_utf-8_mdbg/cedict_ts.U8')
    for sent in data:
        newsent = []
        for word in sent:
            if re.search('[a-zA-Z]',word) is not None:
                try: w = eng.transliterate(unicode(word,'utf-8'))
                except UnicodeDecodeError: w = ''
            else:
                try: w = chi.transliterate(unicode(word,'utf-8'))
                except UnicodeDecodeError: w = ''
            newsent.extend([x for x in w])
        phonData.append(newsent)
    # Make co-occurrence matrix
    phonData = list(itertools.chain.from_iterable(phonData))
    phonRep, phon_i = co_matrix(phonData)
    # Single-value decomposition so each phon is a 7dim vector
    u, s, vh = np.linalg.svd(phonRep)
    phonRep = u[:,:7]
    np.savetxt("phonRep3.txt", np.array(phonRep))
    np.save('phonI3.npy', phon_i)
else:
    phonRep = np.loadtxt("phonRep3.txt")
    phon_i = np.load('phonI3.npy', allow_pickle='TRUE').item()


# Generate semantic representations 
# Make co-occurrence matrix
print("Generating semantic representations...")
semData = list(itertools.chain.from_iterable(data))
semRep, sem_i = co_matrix(semData)
# Single-value decomposition so each word is a 100dim vector
u, s, vh = np.linalg.svd(semRep)
semRep = u[:,:100]


## Get 400 most frequent words
print("Collecting most frequent words...")
allwords = [word for sent in data for word in sent]
mostCommon = nltk.FreqDist(allwords).most_common(400)


## Setup
print("Initializing SOMs and Hebbian network...")
# Initialize 2 SOMs
# Word form -> lexical/phonological SOM
som1 = Som(matrix_size=50, input_size=56,low=-1)
# Word meaning -> semantic SOM
som2 = Som(matrix_size=50, input_size=100,low=-1)
'''
# Load stored SOM
som1.load('/mnt/c/Users/Tessa/Documents/umass/LING692c_final/50som1.npz')
som2.load('/mnt/c/Users/Tessa/Documents/umass/LING692c_final/50som2.npz')
'''
# Initialize Hebbian network
hub = HebbianNetwork("hub") 

# Add SOMs as nodes
hub.add_node("som1", (50, 56))
hub.add_node("som2", (50, 100))
    
# Connect nodes using Hebbian connections
hub.add_connection(0,1)


## Training
print("Training SOMs and network...")

epochs = 1000
lr = 0.1
r = 10.0
eng = epitran.Epitran('eng-Latn')
chi = epitran.Epitran('cmn-Hant', cedict_file='cedict_1_0_ts_utf-8_mdbg/cedict_ts.U8')

for epoch in range(epochs):
    # Get random input vectors
    i = random.randrange(0,400,1)
    word = mostCommon[i][0]
    pinput = []
    if re.search('[a-zA-Z]',word) is not None:
        try: w = eng.transliterate(unicode(word,'utf-8'))
        except UnicodeDecodeError: continue
    else:
        try: w = chi.transliterate(unicode(word,'utf-8'))
        except UnicodeDecodeError: continue
    for phon in w:
        pindex = phon_i[phon]
        pinput.extend(phonRep[pindex,:])
    if len(pinput) < 56:
        pinput.extend([0.0]*(56-len(pinput)))
    if len(pinput) > 56:
        pinput = pinput[:56]
    pinput = np.array(pinput)
    sindex = sem_i[word]
    sinput = semRep[sindex,:]
    
    # Train SOMs
    bmu_i = som1.return_BMU_index(pinput)
    bmu_n = som1.return_unit_round_neighborhood(bmu_i[0], bmu_i[1],r)
    som1.training_single_step(pinput, bmu_n, lr, r)
    
    bmu_i = som2.return_BMU_index(sinput)
    bmu_n = som2.return_unit_round_neighborhood(bmu_i[0], bmu_i[1],r)
    som2.training_single_step(sinput, bmu_n, lr, r)

    # Train connections
    som1_act = som1.return_activation_matrix(pinput)
    som2_act = som2.return_activation_matrix(sinput)
    hub.set_node_activations(0, som1_act)
    hub.set_node_activations(1, som2_act)
    
    hub.learning(learning_rate=0.1, rule="hebb")

    if epoch%25==0:
        print("\nEpoch: " + str(epoch))

# Save the network
som1.save('/mnt/c/Users/Tessa/Documents/umass/LING692c_final/', '50som1')
som2.save('/mnt/c/Users/Tessa/Documents/umass/LING692c_final/', '50som2')


# Label SOMs by mapping input vectors to the best-matching output weights
print("Labeling SOMs...")
eng = epitran.Epitran('eng-Latn')
chi = epitran.Epitran('cmn-Hant', cedict_file='cedict_1_0_ts_utf-8_mdbg/cedict_ts.U8')
labels = []
for each in range(0,400):
    word = mostCommon[each][0]
    pinput = []
    if re.search('[a-zA-Z]',word) is not None:
        try: w = eng.transliterate(unicode(word,'utf-8'))
        except UnicodeDecodeError: continue
    else:
        try: w = chi.transliterate(unicode(word,'utf-8'))
        except UnicodeDecodeError: continue
    for phon in w:
        pindex = phon_i[phon]
        pinput.extend(phonRep[pindex,:])
    if len(pinput) < 56:
        pinput.extend([0]*(56-len(pinput)))
    if len(pinput) > 56:
        pinput = pinput[:56]
    sindex = sem_i[word]
    sinput = semRep[sindex,:]
    labels.append([unicode(word,'utf-8'), som1.return_BMU_index(np.array(pinput)), som2.return_BMU_index(sinput)])
np.savetxt("labels.txt", np.array(labels),fmt='%s')

print("Done!")
