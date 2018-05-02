"""    
    Authors: Victor Mireles  & Artem Revenko for Semantic Web Company
    Cite:
        Mireles V., Revenko A. "Evolution of Semantically Identified Topics"
        CEUR vol 1923 (2017)
        http://ceur-ws.org/Vol-1923/article-06.pdf
"""

import numpy as np
import light_topic_transitions.optimization_topic_matching as otm
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD as SVD

from light_topic_transitions import matrix_decomposition_utils as mdu
from urllib.parse import quote




def decompose_into_topics(document_concept_matrix,
                          k_estimator=mdu.estimate_k_singular,
                          l1_ratio=1,
                          alpha=1,
                          sparsity_scale=0.15,
                          decompo='NMF',
                          freq_threshold=0.85,
                          normalize=True,
                          mink=4):

    # We ignore concepts that are present too many times in this matrix
    # Or not present at all. Prescence is binary.
    binz = np.zeros_like(document_concept_matrix)
    binz[document_concept_matrix > 0] = 1
    freqs = binz.sum(axis=0)
    num_docs = document_concept_matrix.shape[0]
    bad_cpts = np.nonzero(freqs > freq_threshold * num_docs)[0].tolist()
    bad_cpts += np.nonzero(freqs <= 0)[0].tolist()
    good_cpts = range(document_concept_matrix.shape[1])
    good_cpts = [c for c in good_cpts if (c not in bad_cpts)]

    m_clean = document_concept_matrix[:, good_cpts]

    if isinstance(k_estimator, int):
        k = k_estimator
    else:
        k, gaps, deltas = k_estimator(m_clean, exp=10)
    k = max([k, np.ceil(num_docs / 100), mink])
    print("\tk=" + str(k), end=" ")
    if decompo == 'NMF':
        model = NMF(n_components=int(k),
                    init='random',
                    random_state=0,
                    l1_ratio=l1_ratio,
                    alpha=alpha)
    else:
        model = SVD(n_components=int(k))

    # print("\tC:", m_clean.shape)
    tdm = model.fit_transform(m_clean)
    topic_concept_matrix, threshold = mdu.remove_excess_nonzero(
        m_clean,
        model,
        good_cpts,
        numorigcpts=document_concept_matrix.shape[1],
        scale=sparsity_scale)
    tcm, tdm = mdu.remove_blank_topics(topic_concept_matrix, tdm)

    if normalize:
        rowsums = tcm.sum(axis=1)
        for top in range(tcm.shape[0]):
            tcm[top, :] = tcm[top, :] / rowsums[top]

    return tdm, tcm


def get_errors(m, min_k=2, max_k=40):
    errors = []
    for k in range(min_k, max_k+1):
        model = NMF(n_components=k,
                    init='random',
                    random_state=0,
                    l1_ratio=1,
                    alpha=1)
        model.fit(m)
        print(k, end=" ", flush=True)
        errors.append(model.reconstruction_err_)

    return errors


def find_transitions_base(m1,
                          m2,
                          k_estimator,
                          l1_ratio,
                          alpha,
                          sparsity_scale,
                          decompo='SVD',
                          remove_overrepresented=True,
                          freq_threshold = 0.4):

    topic_concept_lists = []
    topic_concept_matrices = []
    topic_document_matrices = []
    good_cpts = range(m1.shape[1])
    if remove_overrepresented:
        # print("ignoring concepts present in more than",
        #       freq_threshold,
        #       "of all documents")
        bothm = np.vstack((m1, m2))
        bothm[np.nonzero(bothm)] = 1
        freqs = bothm.sum(axis=0)
        bad_cpts = np.nonzero(freqs > freq_threshold*bothm.shape[0])[0].tolist()
        bad_cpts += np.nonzero(freqs <= 0)[0].tolist()
        good_cpts = [c for c in good_cpts if (c not in bad_cpts)]

    for im, m in enumerate([m1, m2]):

        dtm, topic_concept_matrix = decompose_into_topics(m)

        topic_concept_lists.append(
            list_concepts_per_topic(topic_concept_matrix))
        topic_concept_matrices.append(topic_concept_matrix)
        topic_document_matrices.append(dtm)
        # print("\t", k, "->", topic_concept_matrix.shape, tdm.shape)

    return topic_concept_lists,\
        topic_document_matrices,\
        topic_concept_matrices


def find_transitions_jaccard(m1,
                             m2,
                             k_estimator=mdu.estimate_k_transition,
                             l1_ratio=1,
                             alpha=1,
                             sparsity_scale=0.5,
                             decompo='NMF'):

    topic_concept_lists, topic_document_matrices, topic_concept_matrices = \
      find_transitions_base(m1=m1,
                            m2=m2,
                            k_estimator=k_estimator,
                            l1_ratio=l1_ratio,
                            alpha=alpha,
                            sparsity_scale=sparsity_scale,
                            decompo=decompo)

    jacc_matrix = np.zeros((len(topic_concept_lists[0]),
                            len(topic_concept_lists[1])))
    for i in range(jacc_matrix.shape[0]):
        for j in range(jacc_matrix.shape[1]):
            ja = jacc_coeficient(topic_concept_lists[0][i],
                                 topic_concept_lists[1][j])
            jacc_matrix[i, j] = ja

    return topic_document_matrices, topic_concept_matrices, jacc_matrix


def find_transitions_optimization(m1,
                                  m2,
                                  k_estimator=mdu.estimate_k_transition,
                                  l1_ratio=1,
                                  alpha=1,
                                  sparsity_scale=0.5,
                                  decompo='NMF',
                                  binarize=False,
                                  remove_overrepresented=True,
                                  normalize=True):

    topic_concept_lists, topic_document_matrices, topic_concept_matrices = \
        find_transitions_base(m1=m1,
                              m2=m2,
                              k_estimator=k_estimator,
                              l1_ratio=l1_ratio,
                              alpha=alpha,
                              sparsity_scale=sparsity_scale,
                              decompo=decompo,
                              remove_overrepresented=remove_overrepresented)

    m_first = topic_concept_matrices[0].copy()
    m_second = topic_concept_matrices[1].copy()
    if normalize:
        for mat in [m_first, m_second]:
            rowsums = mat.sum(axis=1)
            for top in range(mat.shape[0]):
                mat[top, :] = mat[top, :] / rowsums[top]

    if binarize:
        m_first[np.nonzero(m_first)] = 1
        m_second[np.nonzero(m_second)] = 1

    m = otm.get_topic_transition(m_first,
                                 m_second)

    return topic_document_matrices, topic_concept_matrices, m.T


def list_concepts_per_topic(topic_concept_matrix):
    list_per_topic = []
    for topic in range(topic_concept_matrix.shape[0]):
        list_per_topic.append(np.nonzero(topic_concept_matrix[topic, :])[0])

    return list_per_topic


def jacc_coeficient(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    union = set_1 | set_2
    if len(union) == 0:
        return 0
    else:
        return len(set_1 & set_2) / len(union)

