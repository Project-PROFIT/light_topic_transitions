"""    
    Authors: Victor Mireles  & Artem Revenko for Semantic Web Company
    Cite:
        Mireles V., Revenko A. "Evolution of Semantically Identified Topics"
        CEUR vol 1923 (2017)
        http://ceur-ws.org/Vol-1923/article-06.pdf
"""

import topic_transition_calculator as ttc
import random as ran
import datetime


class sample_object:
    def __init__(self, cpts):
        self.cpts = cpts


# Generate artificial data  ---------------------

num_lists = 15
num_objs_per_list = 170
mean_cpts_per_obj = 25
p_id = "profit"
prefix = "http://pro.fit/"
date_0 =  datetime.datetime.now()
delta_t = datetime.timedelta(days=10)
dates = [str((date_0 + i*delta_t).date()) for i in range(num_lists)]


whole_vocab, whole_labels = ttc.fetch_whole_vocab(p_id,
                                                  prefix,
                                                  path="./")

whole_vocab = whole_vocab[1:100]
whole_labels = {x:whole_labels[x] for x in whole_vocab}
lists = []
for i in range(num_lists):
    this_list = []
    for o in range(num_objs_per_list):
        this_dict = {}
        num_cpts = int(max([2, mean_cpts_per_obj + ran.uniform(-10, 10)]))
        cpts = ran.sample(whole_vocab, num_cpts)
        cpt_dict = {c: ran.uniform(1, 5) for c in cpts}
        this_o = sample_object(cpt_dict)
        this_list.append(this_o)
    lists.append(this_list)


js = ttc.jsons_for_several_lists_of_doc_objects(lists,
                                                project_id=p_id,
                                                prefix=prefix,
                                                dates=dates)



import json;
fn = "../topic_visualization/martins/json/response_2010-03-01_to_2011-02-07.json";
json.dump(js, open(fn, "wt"))


