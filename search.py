#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !rm -rf /ssd_scratch/cvit/aryamaan/
# !mkdir /ssd_scratch/cvit/aryamaan/
# !scp aryamaanjain@ada.iiit.ac.in:/share3/aryamaanjain/ire/files.zip /ssd_scratch/cvit/aryamaan/
# !unzip -q /ssd_scratch/cvit/aryamaan/files.zip -d /ssd_scratch/cvit/aryamaan/


# In[ ]:


import numpy as np
import re
import Stemmer
from multiprocessing import Pool
import time
import os


# In[ ]:


def give_tokens(query):
    categories = []
    
    if re.search('[tibclr]:', query):
        break_loc = [(m.start(0), m.end(0)) for m in re.finditer('[tibclr]:', query)]
        N = len(break_loc)
        
        if break_loc[0][0] != 0:
            category = ['p', query[:break_loc[0][0]]]
            categories.append(category)
    
        for i in range(N):
            category = []
            category.append(query[break_loc[i][0]])
            l = break_loc[i][1]
            r = break_loc[i+1][0] if i+1<N else len(query)
            category.append(query[l:r])
            categories.append(category)
    
    else:  # plain query
        categories.append(['p', query])
        
    tokens = []
    
    for cat in categories:
        label = cat[0]
        lower = cat[1].lower()
        split = re_tokenize.split(lower)
        stems = stemmer.stemWords(split)
        for stem in stems:
            if len(stem) > 2:
                tokens.append([label, stem])
                
    return tokens


# In[ ]:


def calculate_score(arg):
    category, N, count, page_id = arg 
    body       = count & 255
    title      = 1 if count & (1<<8)  > 0 else 0
    infobox    = 1 if count & (1<<9)  > 0 else 0
    categories = 1 if count & (1<<10) > 0 else 0
    references = 1 if count & (1<<12) > 0 else 0
    links      = 1 if count & (1<<11) > 0 else 0
    
    body = np.log2(1+body)
    
    c_impt = 10
    if   category == 'b':
        body *= c_impt
    elif category == 't':
        title *= c_impt
    elif category == 'i':
        infobox *= c_impt
    elif category == 'c':
        categories *= c_impt
    elif category == 'r':
        references *= c_impt
    elif category == 'l':
        links *= c_impt

    tf = body + 20*title + 5*infobox + 5*categories + 2*references + 2*links
    
    c_idf = 1
    idf = np.log2(1+c_idf/N)
    score = tf*idf
    
    return score, page_id


# In[ ]:


def give_scores(query):
    tokens = give_tokens(query)
    scores = {}

    for category, token in tokens:
        folder = token[:3]
        folder  = root_inverted_index + folder
        if not os.path.isdir(folder):
            continue
        
        posting_list = np.load(folder + '/posting_list.npy')
        index_offset = np.load(folder + '/index_offset.npy')
        index_word   = np.loadtxt(folder + '/index_word', dtype='str')
        
        word_loc = np.searchsorted(index_word, token)
        pl_start = index_offset[word_loc-1] if word_loc > 0 else 0
        pl_stop  = index_offset[word_loc]
        pl_query = posting_list[pl_start:pl_stop]
        
        pl_query_len = len(pl_query)
        max_qlen = 60000
        if pl_query_len > max_qlen:
            ind = np.random.choice(pl_query_len, max_qlen, replace=False)
            pl_query = pl_query[ind]
            pl_query_len = 60000
        
        print(pl_query_len)
        
#         for page_id, count in pl_query:
#             score = calculate_score(category, pl_query_len, count)
#             if page_id in scores:
#                 scores[page_id] += score
#             else:
#                 scores[page_id]  = score


        with Pool(num_cores) as p:
            args = [(category, pl_query_len, count, page_id) for page_id, count in pl_query]
            score_pid = p.map(calculate_score, args)
                
        for score, page_id in score_pid:
            if page_id in scores:
                scores[page_id] += score
            else:
                scores[page_id]  = score

    return scores


# In[ ]:


def give_best_pages(scores, titles):
    scores = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)
    
    best_pages = []
    for i, (page_id, score) in enumerate(scores):
        if i==10:
            break
        best_pages.append([page_id]) 
        
    id_offset = np.load(root_files + 'id_offset.npy')
        
    id_loc = np.searchsorted(id_offset[:,0], best_pages[0][0])
    offset_r = id_offset[id_loc,1]
    offset_l = id_offset[id_loc-1,1] if id_loc > 0 else 0
    
    for i in range(len(best_pages)):
        page_id = best_pages[i][0]
        id_loc = np.searchsorted(id_offset[:,0], page_id)
        offset_r = id_offset[id_loc,1]
        offset_l = id_offset[id_loc-1,1] if id_loc > 0 else 0
        
        # handle.seek(0)
        # handle.seek(offset_l)
        # title = handle.read(offset_r-offset_l)
        title = titles[offset_l:offset_r]
        best_pages[i].append(title)
    
    return best_pages


# In[ ]:


# root_files = '/ssd_scratch/cvit/aryamaan/files/'
# root_files = '/home/aryamaanjain/ire/phase_2/files/'
root_files = '/ssd_scratch/cvit/aryamaan/files/'
root_inverted_index = root_files + 'inverted_index/'

re_tokenize = re.compile('[^a-z0-9]')
stemmer = Stemmer.Stemmer('english')

with open(root_files + 'titles', 'r') as handle:
    titles = handle.read()

num_cores = 20


# In[ ]:


queries = [
    'random',
    't:sachin b:hello'
]

for query in queries:
    start = time.time()
    scores = give_scores(query)
    best_pages =  give_best_pages(scores, titles)
    end = time.time()
    
    print(query)
    for i in best_pages:
        print('%12d, %s'%(i[0], i[1]))
    print((end-start), '\n')

