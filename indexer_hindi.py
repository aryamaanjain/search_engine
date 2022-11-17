#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import Stemmer
from collections import Counter
import numpy as np
from nltk import word_tokenize
import pickle


# In[2]:


def process(a):
    a = a.replace('।', ' ')
    a = re_tokenize.split(a)
    
    b = []
    for i in a:
        if i not in stop_words and len(i) > 3:
            b.append(i)
    
    b = stemmer.stemWords(b)
    return b


# In[3]:


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


# In[4]:


def parse_xml():
    global id_title
    filepath = filepath_wiki_dump_small
    page_status = 0
    
    cnt = 0
    with open(filepath, "r") as file:
        for line in file:
            if line == "  <page>\n":
                page_status = 1
                cnt += 1
                print(cnt/284425*100, end='\r')
            elif line == "  </page>\n":  # do processing and variable clearing
                page_status = 0
                
                ref_list = re_reference.findall(body)
                for ref in ref_list:
                    references += " " + ref[11:-12]
                
                page_id = int(page_id)
                id_title[page_id] = title
                title = process(title)
                infobox = process(infobox)
                body = process(body)
                category = process(category)
                links = process(links)
                references = process(references)
                
                add_to_index(title, page_id, infobox, body, category, links, references)
            
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


# In[5]:


def index_to_file(index):
    posting_list = []
    index_word = ""
    index_offset = []
    ctr = 0
    
    for word in index:
        ctr += len(index[word])
        posting_list.extend(index[word])
        index_word += word + " "
        index_offset.append(ctr)
    
    posting_list = np.array(posting_list, dtype=np.uint32)
    index_offset = np.array(index_offset, dtype=np.uint32)
    
    np.save(root + 'posting_list_hindi.npy', posting_list)
    np.save(root + 'index_offset_hindi.npy', index_offset)
    with open(root + 'index_word_hindi', 'w') as handle:
        handle.write(index_word)
        
    with open(root + 'id_title.pickle', 'wb') as handle:
        pickle.dump(id_title, handle)


# In[6]:


stop_words = {'अंदर', 'अत', 'अदि', 'अप', 'अपना', 'अपनि', 'अपनी', 'अपने', 'अभि', 'अभी', 'आदि', 'आप', 'इंहिं', 'इंहें', 'इंहों', 'इतयादि', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकि', 'इसकी', 'इसके', 'इसमें', 'इसि', 'इसी', 'इसे', 'उंहिं', 'उंहें', 'उंहों', 'उन', 'उनका', 'उनकि', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसि', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'एसे', 'ऐसे', 'ओर', 'और', 'कइ', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफि', 'काफ़ी', 'कि', 'किंहें', 'किंहों', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसि', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोइ', 'कोई', 'कोन', 'कोनसा', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जहां', 'जा', 'जिंहें', 'जिंहों', 'जितना', 'जिधर', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जेसा', 'जेसे', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिंहें', 'तिंहों', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थि', 'थी', 'थे', 'दबारा', 'दवारा', 'दिया', 'दुसरा', 'दुसरे', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहिं', 'नहीं', 'ना', 'निचे', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पुरा', 'पूरा', 'पे', 'फिर', 'बनि', 'बनी', 'बहि', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भि', 'भितर', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यहां', 'यहि', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रवासा', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वगेरह', 'वरग', 'वर्ग', 'वह', 'वहाँ', 'वहां', 'वहिं', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभि', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'हि', 'ही', 'हुअ', 'हुआ', 'हुइ', 'हुई', 'हुए', 'हे', 'हें', 'है', 'हैं', 'हो', 'होता', 'होति', 'होती', 'होते', 'होना', 'होने'}
root = '/home/aryamaanjain/ire/phase_2/'
filepath_wiki_dump_small = root + 'hiwiki-20210720-pages-articles-multistream.xml'
re_reference = re.compile("&lt;ref&gt;.*?&lt;/ref&gt;")
re_tokenize = re.compile('[^\u0900-\u097F]')
stemmer = Stemmer.Stemmer('hindi')
index = {}
id_title = {}
parse_xml()
index_to_file(index)


# In[1]:


import pickle
with open('/home/aryamaanjain/ire/phase_2/files_hindi/' + 'id_title.pickle', 'rb') as handle:
    titles = pickle.load(handle)
print(len(titles))

