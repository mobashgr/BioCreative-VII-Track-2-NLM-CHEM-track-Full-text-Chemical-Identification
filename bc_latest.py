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


def parse_xml_recognize_xml():
    torch_device= torch.device("cuda:0") #GP

    print(torch_device)

    model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_chemical_ner")

    tokenizer = AutoTokenizer.from_pretrained(

        "alvaroalon2/biobert_chemical_ner")

    ner_model = pipeline('ner',

                         grouped_entities=True,

                         use_fast=True,

                         model=model,

                         tokenizer=tokenizer, device=0)

    columns = {'word', 'start', 'end', 'entity_group', 'score'}



    with open('/hits/fast/sdbv/mobashgr/YOLO.xml', 'rb') as f:

        tree = ElementTree.parse(f)
        root = tree.getroot()
        for neighbor in root.iter('passage'):
             offset=neighbor.find('.offset')
             rating=neighbor.find('.text').text              
             counter=-1
             s = (ner_model(rating))
              #  print (s)
              #  print(passagetext)
             df = pd.DataFrame(columns=columns)
             for item in s:
                   df = df.append(item, ignore_index= True)
             if not df.empty:
                        start = df['start'].tolist() 
                        end = df['end'].tolist()
                        span = sorted(start+end)
                        #print("Start","End")
                       # print("\n")
                        #print(span)
                        Finalspan = []
                        for i in span: 
                             if i not in Finalspan and span.count(i) <=1: 
                                Finalspan.append(i)
                             #print("duplicate removal")
                        #print(Finalspan)
                   
                        for i in range(0,len(Finalspan)-1,2):
                            value = Finalspan[i:i+2]
                            if len(value)>0:
                              #  print (value) 
                                t= (rating[value[0]:value[1]]) #start
                                counter+=1
                           # if len (start) and len(end) >0 :
                                Annotation = ElementTree.SubElement(neighbor, 'annotation', {'id':str (counter)})
                              #  print(counter)
                                Typer = ElementTree.SubElement(Annotation, 'infon ',{'key':'type'})
                                Typer.text= 'Chemical'
                                #print(Typer)
                                starrt=int(value[0])
                              #  print(starrt)
                                Span=int(offset.text)+starrt
                              #  print(Span)
                                Location= ElementTree.SubElement(Annotation, 'location',{'offset':str(Span), 'length':str(len(t))})
                                Word=ElementTree.SubElement(Annotation, 'text')
                                Word.text=t
                                print(Word.text)
    tree.write('/hits/fast/sdbv/mobashgr/YOLO_21.10.xml')
    print('DONE')
             
          
       
            
        
print(parse_xml_recognize_xml())

