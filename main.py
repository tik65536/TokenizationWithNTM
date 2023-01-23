import pandas as pd
import numpy as np
import torch
import re
from datasets import load_dataset
import spacy
from ntm import NTM
import math
import argparse
import json


#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")

#https://www.kaggle.com/code/youssefamdouni/movie-review-classification-using-spacy
def clean_review(text):
    clean_text = re.sub('<br\s?\/>|<br>', '', text)
    #clean_text = re.sub('[^a-zA-Z\']', ' ', clean_text)
    #clean_text = clean_text.lower()
    return clean_text

parser = argparse.ArgumentParser()

parser.add_argument('-seqlen', type=int, default=5,
                    help='Seq Len for reading token')

parser.add_argument('-memsize', type=int, default=10,
                    help='memory size')

parser.add_argument('-controllerDim', type=int, default=100,
                    help='memory size')

parser.add_argument('-head', type=int, default=5,
                    help='head count')
args = parser.parse_args()


imdb_dataset = load_dataset('imdb', split=['train[:10000]+train[15000:]', 'train[10000:15000]', 'test'])
#imdb_dataset = load_dataset('imdb')
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[1]['text']
test_pd["label"]=imdb_dataset[1]['label']
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
test_pd=test_pd.sample(frac=1)

parser = argparse.ArgumentParser()

epoch=100
wv_size=nlp.vocab.vectors_length
seqlen=args.seqlen
controllerDim=args.controllerDim
memorySize=args.memsize
outdim=1
headcount=args.head

print(f'seqlen:{seqlen}, memsize:{memorySize}, controllerDim={controllerDim}, headCount:{headcount}',flush=True)

ntm=NTM(wv_size,outdim,controllerDim,memorySize,wv_size,headcount)

criterion = torch.nn.BCEWithLogitsLoss()
#optimizer = torch.optim.RMSprop(ntm.parameters())
optimizer = torch.optim.Adam(ntm.parameters())

print(f'Start',flush=True)
for step in range(epoch):
    # Sample data
    train_pd=train_pd.sample(frac=1)
    train_pd=train_pd.reset_index()
    docs = nlp.pipe(train_pd['text'])
    pdidx=0
    for doc in docs:
        optimizer.zero_grad()
        ntm.reset()
        doc_tokencount=len(doc)
        output=None
        length=0
        losses = []
        if(seqlen>1):
            output = torch.FloatTensor(doc_tokencount-seqlen, 1)
            length=doc_tokencount-seqlen
        else:
            output = torch.FloatTensor(doc_tokencount, 1)
            length=doc_tokencount
        oidx=0
        for idx in range(0,length):
            print(f'{idx:3d}/{doc_tokencount:5d}',end="\r",flush=True)
            token=[ t for t in doc[idx:idx+seqlen] ]
            token=[ t.vector for t in token ]
            if(len(token)<seqlen):
                remain=seqlen-len(token)
                for i in range(remain):
                    token.append(np.zeros((wv_size,)))
            input_data=torch.Tensor(np.array(token))
            output[oidx]=ntm(input_data)
            oidx+=1
        print(oidx)
        eraseMedian=np.median(np.array(ntm.eraseHistory),axis=1)
        addMedian=np.median(np.array(ntm.addHistory),axis=1)
        target=torch.full((len(output),1),train_pd['label'][pdidx],dtype=torch.float)
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(ntm.parameters(), 10)
        optimizer.step()
        print(f'Doc# {pdidx:5d}, TokenCount:{doc_tokencount}, Memory load:{ntm.memory.loading():0.3f}, '+
                f'Erase:({np.quantile(eraseMedian,0.25):0.3f},{np.quantile(eraseMedian,0.5):0.3f},:{np.quantile(eraseMedian,0.75):0.3f}), '+
                f'Add:({np.quantile(addMedian,0.25):0.3f},{np.quantile(addMedian,0.5):0.3f},{np.quantile(addMedian,0.75):0.3f}), '+
                f'Loss:{np.mean(losses):0.3f}, Target:{train_pd["label"][pdidx]:1d}, AvgOut:{torch.sigmoid(output).mean():0.3f}',flush=True)
        pdidx+=1
    print(f'Step {step:3d} == Loss {np.mean(losses):.3f}',flush=True)
