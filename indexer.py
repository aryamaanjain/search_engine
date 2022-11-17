#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import Stemmer
from collections import Counter
import numpy as np
import os
import pickle


# In[ ]:


def process(a):
    a = a.lower()
    a = re_tokenize.split(a)
    a = stemmer.stemWords(a)
    
    b = []
    for i in a:
        if i not in stop_words:
            if not i.isalpha() and not i.isdecimal():
                continue
            elif i.isdecimal() and (int(i)<1700 or int(i)>2100 or len(i)!=4):
                continue
            elif len(i) < 3:
                continue
            elif len(i) < 16: 
                b.append(i)
            else:
                b.append(i[:16])

    return b


# In[ ]:


def add_to_index(title, page_id, infobox, body, category, links, references):
    global index
    doc_index = Counter(body)
    
    for word in doc_index:
        doc_index[word] = min(doc_index[word], 255)
    
    mask_title = 1 << 8
    for word in title:
        if word in doc_index:
            doc_index[word] |= mask_title
        else:
            doc_index[word] = mask_title
    
    mask_infobox = 1 << 9
    for word in infobox:
        if word in doc_index:
            doc_index[word] |= mask_infobox
        else:
            doc_index[word] = mask_infobox
    
    mask_category = 1 << 10
    for word in category:
        if word in doc_index:
            doc_index[word] |= mask_category
        else:
            doc_index[word] = mask_category
            
    mask_links = 1 << 11
    for word in links:
        if word in doc_index:
            doc_index[word] |= mask_links
        else:
            doc_index[word] = mask_links
            
    mask_references = 1 << 12
    for word in references:
        if word in doc_index:
            doc_index[word] |= mask_references
        else:
            doc_index[word] = mask_references
    
    for word in doc_index:
        if word in index:
            index[word].append([page_id, doc_index[word]])
        else:
            index[word] = [[page_id, doc_index[word]]]


# In[ ]:


def index_to_intermediate(page_num):
    global index
    index = sorted(list(index.items()))
    
    seg = {}
    for ele in index:
        key = ele[0][:3]
        if key in seg:
            seg[key].append(ele)
        else:
            seg[key] = [ele]

    for key in seg:
        folder = root_inverted_index + key + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder + str(page_num) + '.pkl'        
        with open(filename, 'wb') as handle:
            pickle.dump(seg[key], handle)
            
    index = {}


# In[ ]:


def save_id_title(itm):
    N = len(itm)
    id_offset = np.zeros((N,2), dtype=np.uint32)
    titles = ""
    itm = sorted(list(itm.items()))
    
    ctr = 0
    for i, (page_id, title) in enumerate(itm):
        id_offset[i,0] = page_id
        ctr += len(title) + 1
        id_offset[i,1] = ctr
        titles += title + " "

    np.save(root_files + "id_offset.npy", id_offset)
    with open(root_files + "titles", 'w', encoding='utf8') as handle:
        handle.write(titles)


# In[ ]:


def parse_xml():
    filepath = filepath_wiki_dump_small
    page_status = 0
    pages_per_save = 5000
    # pages_per_save = 20000
    cur_page = 1
    id_title_map = {}

    with open(filepath, "r") as file:
        for line in file:

            if line == "  <page>\n":
                page_status = 1
            elif line == "  </page>\n":  # do processing and variable clearing
                page_status = 0
                
                ref_list = re_reference.findall(body)
                for ref in ref_list:
                    references += " " + ref[11:-12]
                
                page_id = int(page_id)
                id_title_map[page_id] = title

                title = process(title)
                infobox = process(infobox)
                body = process(body)
                category = process(category)
                links = process(links)
                references = process(references)
                
                add_to_index(title, page_id, infobox, body, category, links, references)
                if cur_page % pages_per_save == 0:
                    index_to_intermediate(cur_page)
                cur_page += 1
                
            elif page_status == 1: # read title
                title = line[11:-9]
                page_status = 2
            elif page_status == 2:  # skip ns
                page_status = 3
            elif page_status == 3: # read id
                page_id = line[8:-6]
                page_status = 4
            
            elif page_status == 4 and line[:11] == "      <text":  # text start
                infobox = ""
                body = ""
                category = ""
                links = ""
                references = ""
                
                if line[-8:-1] == "</text>":
                    page_status = 6  # text end
                else:
                    page_status = 5
                    
                    infobox_index = line.find("{{Infobox")
                    if infobox_index != -1:
                        page_status_text = 1
                        infobox += line[infobox_index+9:-1] + " "
                    else:
                        page_status_text = 0  # another variable to keep track inside text tag
            
            elif page_status == 5:
                if page_status_text == 1:  # infobox
                    if line == "}}\n":
                        page_status_text = 0
                    else:
                        infobox += line[2:]
                elif line[:9] == "{{Infobox":
                    infobox += line[9:]
                    page_status_text = 1
                        
                elif line[:11] == "[[Category:":
                    if line[-3:-1] == "]]":
                        category += line[11:-3] + " "
                    else:  # when category ends with </text>
                        category += line[11:-10] + " "
                
                elif line == "==External links==\n" or line == "== External links ==\n":
                    page_status_text = 2
                elif page_status_text == 2:
                    if line == '\n':
                        page_status_text = 0
                    else:
                        links += line
                
                elif line == "==References==\n" or line == "== References ==\n":
                    page_status_text = 3
                elif page_status_text == 3:
                    if line == '\n':
                        page_status_text = 0
                    else:
                        references += line
                
                else:
                    body += line
                
                
                if line[-8:-1] == "</text>":  # keep this last statement, something like category can preceed this 
                    # [[Category:American news websites]]</text>
                    page_status = 6  # text end
                    
    if cur_page % pages_per_save != 1:
        index_to_intermediate(cur_page)
        
    save_id_title(id_title_map)


# In[ ]:


def merge_index(*dicts):
    def flatten(LoL):
        return [e for l in LoL for e in l]

    rtr={k:[] for k in set(flatten(d.keys() for d in dicts))}
    for k, v in flatten(d.items() for d in dicts):
        rtr[k].extend(v)
    return rtr


# In[ ]:


def intermediate_to_final():
    for folder in os.listdir(root_inverted_index):
        to_merge = []
        for file in os.listdir(root_inverted_index+folder):
            filename = root_inverted_index + folder + '/' + file
            with open(filename, 'rb') as handle:
                to_merge.append(dict(pickle.load(handle)))
            os.remove(filename)
        index = sorted(merge_index(*to_merge).items())
        for _, pl in index:
            pl.sort()
        
        posting_list = []
        index_word = ""
        index_offset = []
        ctr = 0
        
        for word, pl in index:
            ctr += len(pl)
            posting_list.extend(pl)
            index_word += word + " "
            index_offset.append(ctr)
        
        posting_list = [tuple(i) for i in posting_list]
        posting_list = np.array(posting_list, dtype=np.dtype('u4, u2'))
        index_offset = np.array(index_offset, dtype=np.uint32)
        
        np.save(root_inverted_index + folder + "/posting_list.npy", posting_list)
        np.save(root_inverted_index + folder + "/index_offset.npy", index_offset)
        with open(root_inverted_index + folder + "/index_word", 'w', encoding='utf8') as handle:
            handle.write(index_word)


# In[ ]:


filepath_wiki_dump_small = '/home/aryamaanjain/ire/phase_1/enwiki-latest-pages-articles17.xml-p23570393p23716197'
root_files = '/home/aryamaanjain/ire/phase_2/files/'
# filepath_wiki_dump_small = '/ssd_scratch/cvit/aryamaan/enwiki-20210720-pages-articles-multistream.xml'
# root_files = '/ssd_scratch/cvit/aryamaan/files/'
root_inverted_index = root_files + 'inverted_index/'
os.mkdir(root_files)
os.mkdir(root_inverted_index)

stemmer = Stemmer.Stemmer('english')
re_tokenize = re.compile('[^a-z0-9]')
re_reference = re.compile("&lt;ref&gt;.*?&lt;/ref&gt;")
stop_words = {'', 'ref', 'all', 'by', 'most', 'onc', 'this', 'br', 'what', 'not', 'link', 'doe', 'over', 'can', 'same', 'edit', 'categori', 'here', 'is', 'org', 'reflist', 'note', 'may', 'each', 'about', 'below', 'display', 'time', 'few', 'archiv', 'shouldn', 'lt', 'do', 'how', 'nor', 'off', 'sourc', 'themselv', 'after', 'year', 'histori', 'your', 'myself', 'his', 'ought', 'hasn', 'com', 'html', 'includ', 'i', 'let', 'cannot', 'should', 'until', 'of', 'too', 'inlin', 'www', 'have', 'cite', 'both', 'to', 'style', 'whi', 'into', 'isn', 'onli', 'websit', 'on', 'through', 'herself', 'last', 'citi', 'descript', 'defaultsort', 'where', 'in', 'himself', 'shan', 'when', 'access', 'wikipedia', 'my', 'didn', 'unit', 'again', 'becaus', 'an', 'record', 'at', 'list', 'and', 'then', 'hadn', 'id', 'php', 'mustn', 'web', 'text', 'befor', 'caption', 'no', 'page', 'me', 'own', 'type', 'which', 'articl', 'were', 'label', 'haven', 'author', 'there', 'abov', 'could', 'those', 'has', 'are', 'file', 'wasn', 'such', 'yourself', 'had', 'am', 'so', 'itself', 'down', 'against', 'from', 'imag', 'these', 'out', 'you', 'who', 'jpg', 'refer', 'follow', 'a', 'if', 'whom', 'under', 'url', 'up', 'weren', 'did', 'that', 'would', 'live', 'publish', 'he', 'be', 'her', 'state', 'class', 'languag', 'size', 'dure', 'their', 'she', 'titl', 'with', 'common', 'ani', 'our', 'ourselv', 'was', 'name', 'some', 'doesn', 'the', 'further', 'gt', 'wouldn', 'coord', 'use', 'other', 'yourselv', 'they', 'aren', 'quot', 'we', 'http', 'or', 'been', 'more', 'won', 'date', 'but', 'while', 'result', 'https', 'stub', 'also', 'them', 'new', 'it', 'than', 'accessd', 'veri', 'for', 'him', 'amp', 'see', 'between', 'don', 'as', 'control', 'couldn'}
index = {}

parse_xml()
intermediate_to_final()

