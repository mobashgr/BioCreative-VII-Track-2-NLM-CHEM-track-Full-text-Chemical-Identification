
"""

@author: mobashgr- Ghadeer Mobasher
"""

import pandas as pd

import os
import sys
import torch
from pathlib import Path
import glob
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from json2xml import json2xml
from json2xml.utils import readfromurl, readfromstring, readfromjson
from xml.etree import ElementTree
from xml.dom import minidom
import sys

import argparse


def parse_args():
     ChemNER = argparse.ArgumentParser(  description="ChemNER for XML files")
     ChemNER.add_argument(
        "--xmlfilepath",
        type=str,
        help="Path to the XML to be annotated.",
        required=True,
    )
     ChemNER.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
     ChemNER.add_argument(
        "--outputfilepath",
        type=str,
        help="Path to the output of the annotated xml file",
        required=True,
    )
     args = ChemNER.parse_args()
     return args


def main():
    args = parse_args()

    torch_device= torch.device("cuda:0") #GP

    print(torch_device)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    ner_model = pipeline("ner", model=model,
                   tokenizer=tokenizer,aggregation_strategy="max",grouped_entities=True,

                         use_fast=True,device=0)

    columns = {'word', 'start', 'end', 'entity_group', 'score'}



    with open(args.xmlfilepath, 'rb') as f:

        tree = ElementTree.parse(f, ElementTree.XMLParser(encoding="UTF-8"))
        root = tree.getroot()
        for neighbor in root.iter('passage'):
             offset=neighbor.find('.offset')
             rating=neighbor.find('.text')
             if(rating is not None):
                 rating=rating.text
             else:
                 rating=""
             counter=-1
             s = (ner_model(rating))
             print (s)
             df = pd.DataFrame(columns=columns)
             for item in s:
                   df = df.append(item, ignore_index= True)
             if not df.empty:
                        start = df['start'].tolist()
                        end = df['end'].tolist()
                        span = sorted(start+end)
                        Finalspan = []
                        for i in span:
                             if i not in Finalspan and span.count(i) <=1:
                                Finalspan.append(i)
                             #print("duplicate removal")
                        #print(Finalspan)

                        for i in range(0,len(Finalspan)-1,2):
                            value = Finalspan[i:i+2]
                            if len(value)>0:
                                t= (rating[value[0]:value[1]]) #start
                                counter+=1
                                Annotation = ElementTree.SubElement(neighbor, 'annotation', {'id':str (counter)})
                                Typer = ElementTree.SubElement(Annotation, 'infon',{'key':'type'})
                                Typer.text= 'Chemical'
                                starrt=int(value[0])
                        
                                Span=int(offset.text)+starrt
                             
                                Location= ElementTree.SubElement(Annotation, 'location',{'offset':str(Span), 'length':str(len(t))})
                                Word=ElementTree.SubElement(Annotation, 'text')
                                Word.text=t
                              
    tree.write(args.outputfilepath)
    print('DONE')

if __name__ == "__main__":
    main()


