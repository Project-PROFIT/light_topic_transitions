"""    
    Authors: Victor Mireles  & Artem Revenko for Semantic Web Company
    Cite:
        Mireles V., Revenko A. "Evolution of Semantically Identified Topics"
        CEUR vol 1923 (2017)
        http://ceur-ws.org/Vol-1923/article-06.pdf
"""

import numpy as np
import scipy
from scipy.ndimage import morphology



def make_probability_matrix(matrix):
    P = np.dot(matrix, matrix.T)
    good_rows = np.ones(P.shape[0])
    for x in range(P.shape[0]):
        sx = P[x, :].sum()
        if sx < 0.0000001:
            good_rows[x] = 0
            continue
        P[x, :] = P[x, :]/sx

    P = P[good_rows== 1,:]
    P = P[:, good_rows == 1]

    return P


def estimate_k_transition(matrix, num_gaps_beyond=None, exp=0, max_k=40):
    P = make_probability_matrix(matrix=matrix)
    #print("\t----", P.shape)
    D = np.diag(P.diagonal()).copy()
    P = P - D
    srow = P.sum(axis=1)
    #print(srow)
    P = P - np.diag(srow.copy())
    #print(len(P.nonzero()[0])/(P.shape[1]*P.shape[0]))
    if exp < 0:
        P = scipy.linalg.expm(P);
    if num_gaps_beyond == None:
        num_gaps_beyond = 2+int(P.shape[0]/5)
    EV = np.linalg.eigvals(P)
    EV.sort()
    EV = EV[::-1]
    EV = EV-EV.max()
    EV = EV[1:]
    gaps = EV[:-1] - EV[1:]
    # PP.figure("eigs")
    # PP.subplot(3, 1, 1)
    # PP.plot(EV[:100], '.-')
    # PP.subplot(3, 1, 2)
    # PP.plot(gaps[2:100])
    delta_beyond = np.zeros(len(gaps))
    for k in range(2, min(int(len(gaps)/2), max_k+1)):
        gaps_beyond = gaps[k + 1:k + num_gaps_beyond]
        delta_beyond[k] = (gaps[k]-gaps_beyond.mean())/gaps_beyond.std()
        if delta_beyond[k] >= delta_beyond.max()/3:
            best_k = k

    best_k = np.argmax(delta_beyond[2:]) + 3
    #best_k = np.max(np.nonzero(gaps[1:max_k]>gaps[1:max_k].max()*0.5)[0]) + 2
    #best_k = np.argmax(gaps[3:100])+4
    # PP.subplot(3, 1, 3)
    # PP.plot(delta_beyond)

    return best_k, gaps, delta_beyond


def estimate_k_singular(matrix, exp, radius=3, max_k=40, min_k=2):
    """
    Computes the eigenvalues of a matrix and uses the "gap" criterion to 
    estimate the optimal dimension in which to express the points that M encodes
    as columns.
    :param matrix: 
    :return: 
    """

    u, s, v = np.linalg.svd(matrix)
    s = s[:int(9 * len(s) / 10)]
    max_gap = 0
    best_k = -1

    #gaps = s[:-radius]-s[radius:]
    #gaps = gaps[:-1]/gaps[1:]
    gaps = []
    for i in range(radius, min([len(s)-2*radius, max_k])):
        gaps.append(s[i-radius] - s[i+radius])
    gaps = np.array(gaps)
    gaps = gaps[:-1] / gaps[1:]

    # If possible, look only in the first third of the gaps
    # Otherwise, in the first half
    # Otherwise everywhere
    denominators = [3, 2, 1]
    for den in denominators:
        num_gaps = int(len(gaps)/den)
        if num_gaps >= 3:
            break
    if num_gaps >= 3:
        look_for_gaps = gaps[2:num_gaps]
        best_k = look_for_gaps.argmax()+2+radius
    else:
        best_k = 1

    return max([best_k, min_k]), gaps, s


def estimate_k_delta_singular(matrix,exp,min_k=6):

    u, s, v = np.linalg.svd(matrix)
    s = s[:int(9 * len(s) / 10)]
    max_gap = 0
    best_k = -1
    gaps = s[:-1]-s[1:]
    delta_gap = gaps[:-1]-gaps[1:]
    best_k = np.argmax(delta_gap[1:])+2
    best_k = max(
        np.nonzero(delta_gap > delta_gap.mean() + delta_gap.std())[0]+[min_k]) + 1

    return best_k, delta_gap, gaps



def plot_document_term(tf,
                       vocab,
                       cmps,
                       mapks=[(0,True),(1,True)],
                       titles=['Leaves','Broaders-of-Leaves'],
                       num_docs=None):

    import random as ran
    import matplotlib.pylab as P

    for i,m in enumerate(mapks):
        M = tf.create_matrix(vocab, cmps[m])
        s = M.sum(axis=0)
        #M = M[:, s>1]

        indices = list(range(M.shape[0]))
        ran.shuffle(indices)
        if num_docs is not None:
            M = M[indices[:num_docs], :]
        nz0 = np.nonzero(M)
        #P.subplot(1, len(mapks), i+1)
        P.scatter(nz0[0], nz0[1], s=0.05);
        P.xlabel('Documents');
        P.ylabel('Concepts');
        P.title(titles[i])





def remove_excess_nonzero(m, model, goodcpts, numorigcpts, scale=0.8):
    w = model.components_
    h = model.transform(m)

    num_entries = m.shape[0]*m.shape[1]
    density_m = len(m.nonzero()[0])/num_entries

    def objective_function(threshold):
        th_matrix = w.copy()
        th_matrix[th_matrix < threshold] = 0
        m_estimate = np.dot(h, th_matrix)
        density_estimate = len(m_estimate.nonzero()[0]) / num_entries
        return np.abs(density_estimate - density_m)

    thrs = scipy.optimize.minimize_scalar(objective_function,
                                          method='bounded',
                                          bounds=(0, 0.8*w.max()))

    threshold = scale * thrs.x
    th_matrix = w.copy()
    tcmatrix = np.zeros((th_matrix.shape[0], numorigcpts))
    if scale > 0:
        th_matrix[th_matrix < threshold] = 0
    tcmatrix[:, goodcpts] = th_matrix
    return tcmatrix, threshold


def remove_blank_topics(topic_concept_matrx,
                        tdm,
                        threshold=0.0001,
                        min_cpts=2):
    thmatrix = np.zeros_like(topic_concept_matrx)
    thmatrix[topic_concept_matrx > threshold] = 1
    sumMtrx = thmatrix.sum(axis=1)

    thmatrix = topic_concept_matrx[sumMtrx >= min_cpts, :].copy()
    thmatrx2 = tdm[:, sumMtrx >= min_cpts].copy()
    if thmatrix.shape[0]*thmatrix.shape[1] == 0:
        return topic_concept_matrx, tdm

    return thmatrix, thmatrx2


def sort_topic_by_concept_blocks(topic_concept_matrx,
                                 min_block_size=3):
    str_element1 = [1 for i in range(min_block_size)]
    str_element2 = [1 for i in range(min_block_size+1)]
    num_topics, num_ctps = topic_concept_matrx.shape
    smallest_concept = np.zeros(num_topics)
    for to in range(num_topics):
        footprint = np.zeros(num_ctps)
        footprint[topic_concept_matrx[to, :] > 0] = 1
        footprint = morphology.binary_dilation(footprint, str_element1)
        footprint = morphology.binary_erosion(footprint, str_element2)
        footprint = footprint.astype(np.int8)
        if len(np.nonzero(footprint)[0]) > 0:
            smallest_concept[to] = np.nonzero(footprint)[0].min()
        else:
            smallest_concept[to] = np.inf

    permutation = np.argsort(smallest_concept)
    topic_concept_matrx = topic_concept_matrx[permutation , :]
    return topic_concept_matrx




