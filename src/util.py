# -*- coding: utf-8 -*-

import sys, codecs
import numpy as np

import torch
import transformers

def parse_vertical(path, drop_punct=True):
    """Yield sentences as list[(surface, lemma, pos)]."""
    sent, in_sent = [], False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("<s"):
                sent, in_sent = [], True
                continue
            if line.startswith("</s"):
                if sent:
                    yield sent
                sent, in_sent = [], False
                continue
            if line.startswith("<"):           # <doc>, <text>, <p>, <block>, ...
                continue
            if not in_sent:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            surface, lemma, tag = cols[0], cols[1], cols[2]
            pos = tag[0] if tag else "X"
            if drop_punct and pos == "Z":
                continue
            sent.append((surface, lemma, pos))
    if sent:                                   # flush if file ends without </s>
        yield sent

def parse_conllu(path, drop_punct=True):
    """Yield sentences as list[(surface, lemma, upos)] from a CoNLL-U file.

    Skips comments (#...), multiword-token rows (ID with '-'), and empty nodes
    (ID with '.'). Falls back to FORM if LEMMA is '_'.
    Column layout: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC.
    """
    sent = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:                       # blank line = sentence boundary
                if sent:
                    yield sent
                sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id, form, lemma, upos = cols[0], cols[1], cols[2], cols[3]
            if "-" in tok_id or "." in tok_id:   # multiword / empty node
                continue
            if drop_punct and upos == "PUNCT":
                continue
            if lemma == "_":                   # unknown lemma
                lemma = form
            sent.append((form, lemma, upos))
    if sent:
        yield sent


def detect_format(path):
    p = path.lower()
    if p.endswith((".conllu", ".conll-u", ".conll")):
        return "conllu"
    return "vertical"

def load_sentences(corpus_file, cased=False):
    """
    Parameters:
    ----------
    corpus_file:str

    cased:bool
        whether or not to consider upper/lower cases
        if False, case is ignored when the target phrase is searched
    

    Rreturn:
    -------=
    sentences: list[list[str]]
        words in sentences. That is, note that sentences are split into words
        here.
    """

    sentences = []
    with codecs.open(corpus_file,  'r', 'utf-8', 'ignore') as fp:
        for sentence in fp:
            sentence = sentence.rstrip()
            if cased  == False:
                sentence = sentence.lower()
            tokens = sentence.split(' ')
            sentences.append(tokens)

    return sentences


def load_kappa_freq_file(filename,
                         ignore_mid_subwords=True,
                         ignore_digit=False,
                         freq_threshold=0,
                         delim='\t'):
    data = []
    with codecs.open(filename,  'r', 'utf-8', 'ignore') as fp:
        for line in fp:
            line = line.rstrip()
            token, kappa, freq = line.split(delim)
            kappa = float(kappa)
            freq = int(freq)
            if token == '[CLS]':
                continue
            if ignore_mid_subwords==True and token.startswith('##'):
                continue
            if ignore_digit==True and token.isdigit():
                continue
            if freq > freq_threshold and kappa > 0:
                data.append((token, kappa, freq))

    return data


def tokenize_and_sum_vectorize(vectorizer, tokenizer, batched_sentences, 
                               start_layer, end_layer, 
                               is_split_into_words=True,
                               device='cpu'):
    """
    To tokenize and vectorize batched sentences. The obtained vectors are all
    normalized so that their norms equal one. This is for the von Mises-Fisher
    distribution.


    Parameters
    ----------
    vectorizer: vectorizer (mostly, BERT-based)

    tokenizer: tokenizer that is compatible with the vectorizer.

    start_layer: int
        selects which layer to use as word vectors (starting layer)

    end_layer: int
        selects which layer to use as word vectors (end layer)
        

    Returns 
    -------
    token2suM_vec: {str:list[numpy array]} 
        dict. mapping token to its summed vectors
    
    """

    vectorizer.to(device)
    vectorizer.eval()
    token2sum_vec = {}
    token2freq = {}
    with torch.no_grad():
        for sentences in batched_sentences:
            tokens = tokenizer(sentences,
                               is_split_into_words=is_split_into_words,
                               return_tensors='pt',
                               padding=True,
                               truncation=True)
                       
            token_ids = tokens['input_ids'].to(device)
            mask_ids = tokens['attention_mask'].to(device)

            output = vectorizer(token_ids, mask_ids)

            # hidden states from start_layer to end_layer - 1
            hidden_states = output.hidden_states[start_layer:end_layer]

            # (layer, vec_dim, tokens)
            hidden_states = torch.stack(hidden_states, dim=0)
            hidden_states = hidden_states.to('cpu').detach().numpy().copy()

            for batch_idx in range(len(sentences)):
                surface_tokens = tokenizer.convert_ids_to_tokens(token_ids[batch_idx])
                for token_idx, token in enumerate(surface_tokens):
                    layer_vecs = hidden_states[:, batch_idx, token_idx]
                    # normalizing for vMF distribution
                    layer_vecs = layer_vecs/np.linalg.norm(layer_vecs,
                                                           axis=1,
                                                           keepdims=True)
                    mean_vec = np.mean(layer_vecs, axis=0)
                    dim = mean_vec.shape
                    sum_vec = token2sum_vec.get(token, np.zeros(dim))
                    sum_vec += mean_vec
                    token2sum_vec[token] = sum_vec
                    token2freq[token] = token2freq.get(token, 0) + 1

    return token2sum_vec, token2freq


def to_batches(instances, batch_size=32):
    num_batch = len(instances)//batch_size
    batches =\
        [ instances[n*batch_size:(n+1)*batch_size] for n in range(num_batch) ]

    rest = len(instances) - num_batch*batch_size
    if rest>0:
        batches.append(instances[num_batch*batch_size:num_batch*batch_size+rest])

    return batches



def cal_concentration(mean_norm, vector_size=768):
    """
    Calculate the concentration parameter kappa of the von Mises-Fisher
    distribution, which is based on the norm of a vector


    Parameters
    ----------
    norm: float ([0, 1])
        norm of the mean vector, which ranges between 0 and 1 (because
        all vectors are normalized to have norm=1 in the von Mises-Fisher
        distribution. This function uses Banerjee et al. (2005)'s 
        approximation.

    vetcor_size: int
        the dimension of the mean vector (and also all vectors in the space).
        its default is set to 1024 coming from 'bert-large' models


    Returns 
    -------
    kappa: float
        the concentration parameter kappa of the von Mises-Fisher distribution
    
    """
    kappa = -1.0

    if mean_norm < 1.0:
        kappa = mean_norm*(float(vector_size) - mean_norm**2)/(1.0 - mean_norm**2)

    return kappa


def to_ids_with_token_idx(tokenizer, sentences, is_split_into_words=True, device='cpu'):
    """
    This method returns token indices only for words that are NOT split into
    subwords. The returned token idices follow the BERT-indexing system, meaning
    that they are added by one (for the special token [CLS]) from the original
    index (of the origina sentence).

    Parameters
    ----------
    tokenizer: BERT-based tokenizer

    sentences: list[str]
    batched sentences consiting of tokens
    """

    tokens = tokenizer(sentences,
                       is_split_into_words=is_split_into_words,
                       return_tensors='pt',
                       padding=True,
                       truncation=True)
                       
    token_ids = tokens['input_ids'].to(device)
    mask_ids = tokens['attention_mask'].to(device)

    target_token_idx2subword_idx = [ {} for _ in sentences ]
    for batch_i in range(len(sentences)):
        # mapping between subword index to token index in the original sentence
        subword_idx2token_idx = tokens.word_ids(batch_i)
        token_idx2subword_indices = {}
        for subword_i, token_i in enumerate(subword_idx2token_idx):
            if token_i != None:
                subword_indices = token_idx2subword_indices.get(token_i, [])
                subword_indices.append(subword_i)
                token_idx2subword_indices[token_i] = subword_indices

        for token_i, subword_indices in token_idx2subword_indices.items():
            # targeting only tokens consiting of one sub-word
            if len(subword_indices) == 1:   
                target_token_idx2subword_idx[batch_i][token_i] =\
                    subword_indices[0]

    return token_ids, mask_ids, target_token_idx2subword_idx
