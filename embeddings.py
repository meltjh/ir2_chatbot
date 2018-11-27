import torch
import numpy as np

def get_embedding_vectors(embedding_dim):
    embedding_vectors = {}
    with open('glove.6B/glove.6B.{}d.txt'.format(embedding_dim), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            embedding_vectors[word] = np.array(line[1:]).astype(np.float)
    glove_vocab = embeddings_vectors.keys()
    print(glove_vocab)
    return glove_vocab, embedding_vectors

def get_weight_matrix(embedding_vectors, word2id, vocab_size, embedding_dim):
    weights_matrix = np.zeros((matrix_len, embedding_dim))
    for word, id in word2id.w2id.items():
        weights_matrix[id] = embedding_vectors[word]
    return weights_matrix

    # matrix_len = len(word2id.id2w)
    #
    # # print(word2id.w2id)
    # for word, id in word2id.w2id.items():
    #     # print(id, word)
    # # for i, word in enumerate(target_vocab):
    #     # try:
    #
    #     # except KeyError:
    #         # weights_matrix[id] = np.random.normal(scale=0.6, size=(emb_dim, ))
    # print(weights_matrix)
    # for

    # weights_matrix = np.zeros((matrix_len, 300))
    # print(weights_matrix)
    # glove = {w: vectors[word2idx[w]] for w in words}
    #         print(line)
        # word = line[0]
        # words.append(word)
        # word2idx[word] = idx
        # idx += 1
        # vect = np.array(line[1:]).astype(np.float)
        # vectors.append(vect)
