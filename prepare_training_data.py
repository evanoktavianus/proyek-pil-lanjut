"""

File ini berisi kode untuk melakukan persiapan sampling training data dari SCIDOCS:

"""

from google.colab import drive
drive.mount('/content/drive')

import json 
import random
import numpy as np


def get_documents_reff(doc_path):
  """
  Fungsi untuk loading keseluruhan informasi dokumen (judul, abstrak, sitasi)

  Inputs:
  a. path pada file

  Outputs:
  a. refs: variable dengan seluruh informasi dokumen

  """
  
  #variable yang menyimpan informasi dokumen
  ref=[]
  with open(ref_path, 'r') as f:
    refs = json.load(f)
  return refs


def get_doc_test(qrel_path):
  """
  fungsi yang mengembalikan paper_ids dan paper inputs untuk testing
  Input:
  a. path pada file qrel dari SCIDOCS
  
  Output:
  paper_ids, paper_inputs
  """

  with open(qrel_path) as f_in:
    qrels = [line.strip() for line in f_in]

  #set yang menyimpan daftar id dokumen
  paper_ids=set()
  for q in qrels:
    row = q.split(' ')
    paper_ids.add(row[2])
    paper_ids.add(row[0])

  print(len(paper_ids))
  paper_ids=list(paper_ids)

  paper_inputs=[]
  for paper_id in paper_ids:
    paper=refs[paper_id]
    paper_item={}
    paper_item['title']=paper['title']
    paper_item['abstract']=paper['abstract']
    paper_inputs.append(paper_item)
  
  return paper_ids, paper_inputs




def produce_sample(num,corpus,output_folder,test_exclude):
  """
    Fungsi untuk menghasilkan sample
    Input:
    a.num: jumlah sample
    b.corpus: sumber corpus dokumen
    c.url: url untuk menyimpan output file 
  """
  paper_ids_list=[]
  paper_ids_set=set()
  metadata={}

  #ambil ids dari corpus
  print('corpus original length: '+str(len(corpus)))
  for ref in corpus:
    if not ref in test_exclude:paper_ids_list.append(ref) 
  
  print('corpus not in test: '+str(len(paper_ids_list)))

  paper_ids_set=set(paper_ids_list)

  def count_valid_references(paper_id):
    num=0
    for ref in corpus[paper_id]['references']:
      if ref in paper_ids_set:num=num+1
    return num

  paper_ids_has_references_set=set()
  #clean sample hilangkan yang tidak memiliki referensi 
  for ref in corpus:
    if count_valid_references(ref) >0: paper_ids_has_references_set.add(ref)
  
  paper_ids_has_references_list=list(paper_ids_has_references_set)
  print('valid corpus: '+str(len(paper_ids_has_references_list)))

  print(len(paper_ids_has_references_list))
  print(len(paper_ids_list))
  random.shuffle(paper_ids_has_references_list)

  #ambil jumlah sample
  if num>0: paper_ids_sample=paper_ids_has_references_list[0:num]
  if num==0: paper_ids_sample=paper_ids_has_references_list

  #bagi menjadi train, validate, dan test
  if num>0: train_sample, validate_sample, test_sample=np.split(paper_ids_sample,[int(num*0.8),int(num*0.99)])
  if num==0: train_sample, validate_sample, test_sample=np.split(paper_ids_sample,[int(len(paper_ids_has_references_list)*0.80),int(len(paper_ids_has_references_list)*0.99)])

  with open(output_folder+'train.txt', 'w') as f:
    for p in train_sample:
      f.write(p+'\n')
    
  with open(output_folder+'val.txt', 'w') as f:
    for p in validate_sample:
      f.write(p+'\n')

  with open(output_folder+'test.txt', 'w') as f:
    for p in test_sample:
      f.write(p+'\n')

  print('corpus length: '+str(len(corpus)))
  
  #variable untuk menyimpan format data sample sesuai spesifikasi specter
  data={}
  print(id)
  def add_to_metadata(paper_id):
    # print(paper_id+" is "+str(paper_id in paper_ids_set))
    metadata[paper_id]={"paper_id":paper_id,"abstract":corpus[paper_id]['abstract'],"title":corpus[paper_id]['title']}

  for paper in paper_ids_sample:

    #tambah paper pada metadata
    add_to_metadata(paper)

    data[paper]={}
    ref_set=set()
    #ambil sitasi langsung
    for ref in refs[paper]['references']:
      if ref in paper_ids_set: 
        #tambah paper pada citation list
        data[paper][ref]={"count":5}

        #tambah paper pada set of references 
        ref_set.add(ref)
        
        add_to_metadata(ref)

      
      #apabila paper referensi ada pada korpus lanjutkan untuk ambil sitasi level 2
        for second_ref in refs[ref]['references']:
          if not second_ref in ref_set: #cek apakah sudah di sitasi secara langsung
            if second_ref in paper_ids_set: 
              data[paper][second_ref]={"count":1} 
              add_to_metadata(second_ref)

  with open(output_folder+'metadata.json','w') as outfile:
    json.dump(metadata,outfile)

  with open(output_folder+'data.json', 'w') as outfile:
    json.dump(data, outfile)


#MAIN

qrel_file="/content/drive/MyDrive/PIL/test.qrel" #path pada file qrel dari SCIDOCS
test_paper_ids,paper_inputs=get_doc_test(qrel_file)
test_set=set(test_paper_ids)

ref_path="/content/drive/MyDrive/PIL/paper_metadata_view_cite_read.json" #path dari scidocs metadata
cite=get_documents_reff(ref_path)

ref_path="/content/drive/MyDrive/PIL/paper_metadata_recomm.json" #path dari scidocs metadata
recom=get_documents_reff(ref_path)

ref_path="/content/drive/MyDrive/PIL/paper_metadata_mag_mesh.json" #path dari scidocs metadata
mag=get_documents_reff(ref_path)

#kombinasikan metadata dari SCIDOCS
cite.update(recom)
cite.update(mag)
refs=cite

#jalankan fungsi untuk memproduce training data dari dataset SCIDOCS (dalam paper menggunakan 30000 sample)
produce_sample(30000,refs,'/content/drive/MyDrive/proyekPIL/',test_set)