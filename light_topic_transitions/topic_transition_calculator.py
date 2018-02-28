"""    
    Authors: Victor Mireles  & Artem Revenko for Semantic Web Company
    Cite:
        Mireles V., Revenko A. "Evolution of Semantically Identified Topics"
        CEUR vol 1923 (2017)
        http://ceur-ws.org/Vol-1923/article-06.pdf
"""

import numpy as np
import light_topic_transitions.topic_transition_utilities as ttu

from light_topic_transitions import matrix_decomposition_utils as mdu


def read_matrix_for_list_of_objects(list_of_objects,
                                    cpts_key='cpts',
                                    whole_vocab=None,
                                    binarize=False):
    """
    
    :param list_of_objects: 
    :param cpts_key: 
    :param whole_vocab: 
    :param binarize: 
    :return: 
    """
    if not whole_vocab:
        all_cpts_list = set()
        for sri in list_of_objects:
            cpt_count_dict = sri.__dict__[cpts_key]
            for cpt, count in cpt_count_dict.items():
                all_cpts_list.add(cpt)
        all_cpts_list = list(all_cpts_list)
    else:
        all_cpts_list = whole_vocab

    concept_uri_to_id = dict()
    for concept_num, k in enumerate(all_cpts_list):
        concept_uri_to_id[k] = concept_num

    num_docs = len(list_of_objects)
    num_cpts = len(all_cpts_list)

    matrix = np.zeros((num_docs, num_cpts))

    for doc_num, sri in enumerate(list_of_objects):
        cpt_count_dict = sri.__dict__[cpts_key]
        for this_URI, count in cpt_count_dict.items():
            matrix[doc_num, concept_uri_to_id[this_URI]] = count

    if binarize:
        matrix[matrix > 0] = 1

    return matrix, all_cpts_list


def fetch_whole_vocab(project_id, prefix, sep="\t", path="./"):
    whole_vocab = []
    whole_labels = {}
    fn = path+'vocabs/'+project_id+'.vocab'
    a = open(fn)
    for row in a:
        rowsplt = row.strip().split(sep)
        if len(rowsplt) != 2:
            continue
        uri = prefix + rowsplt[0]
        whole_vocab.append(uri)
        whole_labels[uri] = rowsplt[1]

    return whole_vocab, whole_labels


def tt_from_2_lists_of_doc_objects(list1, list2, whole_vocab,
                                   k_finder=mdu.estimate_k_singular,
                                   sparsity_scale=0.8,
                                   l1_regularization_ratio=0.8,
                                   alpha=1.0,
                                   decompo='NMF',
                                   remove_overrepresented=True):

    m1, _ = read_matrix_for_list_of_objects(list1, whole_vocab=whole_vocab)
    m2, _ = read_matrix_for_list_of_objects(list2, whole_vocab=whole_vocab)
    topic_document_matrices, topic_concept_matrices, tt_matrix = \
        ttu.find_transitions_optimization(
            k_estimator=k_finder,
            m1=m1,
            m2=m2,
            l1_ratio=l1_regularization_ratio,
            alpha=alpha,
            sparsity_scale=sparsity_scale,
            decompo=decompo,
            remove_overrepresented=remove_overrepresented)

    return topic_document_matrices, tt_matrix, topic_concept_matrices


def tt_for_several_lists_of_doc_objects(lists, whole_vocab):
    """    
    :param lists: 
    :param whole_vocab: 
    :return: 
    """
    topic_document_matrices = []
    transition_matrices = []
    topic_concept_matrices = []

    for i in range(len(lists)-1):
        print("\nTT:", i)
        l1 = lists[i]
        l2 = lists[i+1]
        tdm, tm, tcm = tt_from_2_lists_of_doc_objects(l1, l2,
                                                      whole_vocab=whole_vocab)
        topic_document_matrices.append(tdm[0])
        transition_matrices.append(tm)
        topic_concept_matrices.append(tcm[0])

    topic_document_matrices.append(tdm[1])
    topic_concept_matrices.append(tcm[1])

    return topic_document_matrices, transition_matrices, topic_concept_matrices


def jsons_for_several_lists_of_doc_objects(lists,
                                           project_id=None,
                                           prefix=None,
                                           path="./",
                                           num_topics_per_week=25,
                                           num_cpts_per_topic=10,
                                           new_thrs=0.2,
                                           dates=None,
                                           ):
    """
    :param lists: is a list of lists of object.
     The objects have a property named 'cpts'
        object['cpts'] is a dictionary of URI : count, 
        where URI is a string and count an integer
        all URI's are prexifed with :param prefix: 
    
    :param path: and :param project_id: specify the location of file  
                path/project_id.vocab
        which contains in every row:  URI\tlabel
        URI's exclude the prefix, label can be any (reasonable) string
        the URIs in the objects in the lists must all be included in this file
        
    :param num_topics_per_week: determines the number of topics that will be 
       exported in the json
    :param num_cpts_per_topic: determines the number of concepts that will be 
        exported in the json, in every topic
    :param new_thrs:  if for a given topic all of its transitions are below 
        this, it is assumed to be new.
    :param dates: is a list of dates the same cardinality as lists
     
    :return: a dictionary containing a json object with the results of topic
        transitions between consecutive lists in :param lists: Can be serialized
        using json.dump
    """

    whole_vocab, whole_labels = fetch_whole_vocab(project_id,
                                                  prefix,
                                                  path=path)

    tdms, tms, tcms = tt_for_several_lists_of_doc_objects(lists, whole_vocab)

    all_dicts = []
    prev_good_cpts = set()
    prev_good_topics = set()
    for i, tm in enumerate(tms):
        all_good_cpts = set()
        new_good_topics = set()
        this_tdm = tdms[i+1]
        this_tcm = tcms[i+1]
        prev_tm = tms[i - 1] if i > 0 else np.zeros((tm.shape[0],
                                                     tm.shape[0]))
        top_doc_m = np.zeros_like(this_tdm)
        top_doc_m[np.nonzero(this_tdm)] = 1


        # ------ Find the topics to be exported --------------------------------
        topic_weights = top_doc_m.mean(axis=0).tolist()
        topic_weights_sorted = top_doc_m.mean(axis=0).tolist()
        if len(topic_weights_sorted) < num_topics_per_week:
            topic_weight_threshold = min(topic_weights_sorted)
        else:
            topic_weights_sorted.sort()
            topic_weights_sorted = topic_weights_sorted[::-1]
            topic_weight_threshold = topic_weights_sorted[
                num_topics_per_week - 1]

        good_topic_pairs = [(tnu, topic_weights[tnu])
                            for tnu in range(tm.shape[1]) if
                            (topic_weights[tnu] >= topic_weight_threshold)
                            or np.argmax(tm[:, tnu]) in prev_good_topics]

        print(len(good_topic_pairs), "Good Topics! ")
        if len(good_topic_pairs) > num_topics_per_week + 2:
            good_weights = [y for (x, y) in good_topic_pairs]
            good_weights.sort()
            good_weights = good_weights[::-1]
            weight_thrs = good_weights[num_topics_per_week+2]
            good_topics = [x for (x, y) in good_topic_pairs if y > weight_thrs]
            # print(">removed topics*", weight_thrs, good_weights)
        else:
            good_topics = [x for (x, y) in good_topic_pairs]
        good_topics = [i for i in range(tm.shape[1])]
        print(len(good_topics), "Good Left! ")
        topic_list = []

        # These are the topics for this time point
        for tnu in range(tm.shape[1]):
            to_cpw = {whole_vocab[i]: this_tcm[tnu, i]
                      for i in np.nonzero(this_tcm[tnu, :])[0]}
            if tnu not in good_topics:
                continue

            # ------ Find the concepts to be exported --------------------------
            most_similar = np.argmax(tm[:, tnu])
            new_good_topics.add(tnu)
            print("\t\t", tnu, most_similar)

            list_cpts = [(whole_labels[x][0], v)
                         for x, v in to_cpw.items()]
            list_cpts.sort(key=lambda x: x[1])
            list_cpts = list_cpts[::-1]
            most_prominent_labels = [x[0] for x in list_cpts[:2]]


            cpt_weights_sorted = list(to_cpw.values())
            if len(cpt_weights_sorted) < num_cpts_per_topic:
                cpt_weight_threshold = min(cpt_weights_sorted)
            else:
                cpt_weights_sorted.sort()
                cpt_weights_sorted = cpt_weights_sorted[::-1]
                cpt_weight_threshold = cpt_weights_sorted[
                    num_cpts_per_topic - 1]
            cpt_weight_threshold = min(cpt_weights_sorted)
            good_cpts = {x for x, v in to_cpw.items()
                         if (v >= cpt_weight_threshold or x in prev_good_cpts)}

            # ------ Build the Json Dictionary --------------------------------
            all_good_cpts = all_good_cpts | good_cpts

            topic_dict = {'topic_id': str(tnu),
                          'weight': topic_weights[tnu],
                          'cpts': {whole_labels[x]: to_cpw[x]
                                   for x in good_cpts},
                          'topic_name': str(tnu)
                          + "_"
                          + ";".join(most_prominent_labels)
                          }
            if i >= 0:
                topic_dict['most_similar_in_previous'] = str(most_similar)
            topic_list.append(topic_dict)

        tm2 = tm.copy()
        tm2[tm < new_thrs] = 0
        strtm = [[str(xX) for xX in bB] for bB in tm2]
        thisdict = {'date': dates[i] if dates else str(i),
                    'id': i,
                    'topics': topic_list,
                    'transition': strtm}
        all_dicts.append(thisdict)
    return all_dicts


