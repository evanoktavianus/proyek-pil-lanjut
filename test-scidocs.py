"""
 File ini berisi kode untuk melakukan baseline evaluation kinerja model meliputi:
 a. SPECTER
 b. BERT
 c. SCIBERT
 d. ALBERT
"""
#instalasi dan importasi requirements
import json
import numpy as np
from tqdm import tqdm
from google.colab import drive
import operator 
import pathlib
import os
!pip install pytrec_eval 
import pytrec_eval
from collections import defaultdict
!pip install transformers
!pip install torch
import torch
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
!pip install pytrec_eval
import pytrec_eval
import pickle 

def get_documents_reff(doc_path):
  """
  Fungsi untuk loading keseluruhan informasi dokumen (judul, abstrak, sitasi)

  Input: 
  doc_path: path pada dokumen SCIDOCS

  Retrun:
  a. refs: variable dengan seluruh informasi dokumen

  """
  
  ref=[]
  with open(ref_path, 'r') as f:
    refs = json.load(f)
  return refs


def get_doc_test(qrel_path):
  """
  Fungsi untuk memproses dokumen test dari SCIDOCS
  
  Input:
  a. qrel_path: path file qrel dari SCIDOCS

  Output: 
  paper_ids: list id dari paper test
  paper_inputs: dictionary dari title dan abstrak
  """
  
  #variable yang menyimpan daftar dokumen untuk test
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


def calculate_embeddings(tokenizer, model, result_path, refs,paper_inputs):
  """
  Fungsi untuk menghitung embeddings menggunakan model dari huggingface untuk baseline evaluation
  input:
  a. tokenizer: string tokenizer huggingface
  b. model: string model huggingface
  c. result_path: path untuk save hasil embeddings
  d. refs: referensi metadata dokumen
  e. paper_inputs: input paper yang akan diproses embeddingsnya

  output:
  outputs: hasil embeddings 
  """
  if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

  outputs=[]
  json_line={}
  model.eval()
  model.to(device)

  with open(result_path, 'w') as outfile:

    i=0

    for paper in tqdm(paper_inputs,position=0, leave=True):
      paper_input=paper['title']+tokenizer.sep_token+(paper.get('abstract') or '')
      tokenized_inputs = tokenizer(paper_input, return_tensors='pt', padding='max_length',truncation=True,max_length=512).to(device)
      output = model(**tokenized_inputs)
      cls_output=output[0][0,0,:]
      cls_cpu=cls_output.cpu().detach().numpy()
      outputs.append(cls_cpu)
      json_line['paper_id']=paper_ids[i]
      json_line['title']=refs[paper_ids[i]]['title']
      json_line['embedding']=cls_cpu.tolist()
      json.dump(json_line, outfile)
      outfile.write('\n')
      i=i+1
  
  return outputs

def load_embeddings_from_jsonl(embeddings_path):
  """Load data embedding dari jsonl
    Argumen:
    embedding_path -- path file jsonl embedding

    Returns:
    embeddings
    """
  embeddings = {}
  with open(embeddings_path, 'r') as f:
        for line in tqdm(f, desc='reading embeddings from file...',position=0, leave=True):
            line_json = json.loads(line)
            embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
  return embeddings

def make_run_from_embeddings(qrel_file, embeddings, run_file, topk=5, generate_random_embeddings=False):
    """Melakukan perhitungan embedding dan jarak dokumen dari file embeddings dan ground truth data
    Argumen:
        qrel_file -- qrel file yang berisi ground truth hubungan sitasi dokumen
        embeddings -- dictionary dari data embeddings
        run_file -- file yang akan berisikan perhitungan jarak dokumen
        topk -- jumlah top nearest neighbors            
    Returns:
        None
    """
    with open(qrel_file) as f_in:
        qrels = [line.strip() for line in f_in]

    papers = defaultdict(list) 

    #dapatkan daftar dokumen dari qrels
    for line in qrels:
        row = line.split(' ')
        papers[row[0]].append(row[2])

    results = []

    missing_queries = 0
    key_error = 0
    success_candidates = 0
    
    for pid in papers:
        try:
            emb_query = embeddings[pid]
        except KeyError:
            missing_queries += 1
            continue
        if len(emb_query) == 0:
            missing_queries += 1
            continue
        emb_candidates = []
        candidate_ids = []
        for idx, paper_id in enumerate(papers[pid]):
            try:
                if generate_random_embeddings:
                    emb_candidates.append(np.random.normal(0, 0.67, 200))
                else:
                    emb_candidates.append(embeddings[paper_id])
                candidate_ids.append(paper_id)
                success_candidates += 1
            except KeyError:
                key_error += 1


        # hitung similaritas dokumen
        emb_query = np.array(emb_query)
        distances = [-np.linalg.norm(emb_query - np.array(e))
                     if len(e) > 0 else float("-inf")
                     for e in emb_candidates]

        distance_with_ids = list(zip(candidate_ids, distances))

        sorted_dists = sorted(distance_with_ids, key=operator.itemgetter(1))
        
        added = set()

        for i in range(len(sorted_dists)):
            # output is in this format: [qid iter paperid rank similarity run_id]
            if sorted_dists[i][0] in added:
                continue
            if i < len(sorted_dists) - topk:
                results.append([pid, '0', sorted_dists[i][0], '0', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            else:
                results.append([pid, '0', sorted_dists[i][0], '1', str(np.round(sorted_dists[i][1], 5)), 'n/a'])
            added.add(sorted_dists[i][0])

    pathlib.Path(run_file).parent.mkdir(parents=True, exist_ok=True)

    with open(run_file, 'w') as f_out:
        for res in results:
            f_out.write(f"{' '.join(res)}\n")
    return missing_queries
    
def qrel_metrics(qrel_file, run_file, metrics=('ndcg', 'map')):
    """Perhitungan performance menggunakan pytrec.
    Arguments:
        qrel_file -- qrel file mengandung truth data
        run_file -- hasil prediksi model
        metrics -- evaluasi metrik dari trec_eval,
                   
    Returns:
        metric_values -- dictionary dari metric values (out of 100), dibulatkan pada dua desimal 
    """
    with open(qrel_file, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(run_file, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics))
    results = evaluator.evaluate(run)

    metric_values = {}
    for measure in sorted(metrics):
        res = pytrec_eval.compute_aggregated_measure(
                measure, 
                [query_measures[measure]  for query_measures in results.values()]
            )
        metric_values[measure] = np.round(100 * res, 2)
    return metric_values

#MAIN

#Baseline evaluation untuk SCIBERT
from transformers import AutoTokenizer,AutoModel
result_path="/content/drive/MyDrive/PIL/scibert-embedding.jsonl"
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_outputs=calculate_embeddings_pooler(scibert_tokenizer, scibert_model, result_path, refs,paper_inputs)
scibert_embedding_path='/content/drive/MyDrive/PIL/scibert-embedding.jsonl'
scibert_run_path='/content/drive/MyDrive/PIL/scibert.run'
qrel_path="/content/drive/MyDrive/PIL/test.qrel"
scibert_embedding = load_embeddings_from_jsonl(scibert_embedding_path)
make_run_from_embeddings(qrel_path,scibert_embedding,scibert_run_path)
scibert_cite_results = qrel_metrics(qrel_path, scibert_run_path, metrics=('ndcg', 'map'))
print(scibert_cite_results)


#Baseline evaluation untuk BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
result_path="/content/drive/MyDrive/PIL/bert-embedding.jsonl"
bert_outputs=calculate_embeddings(bert_tokenizer, bert_model, result_path, refs,paper_inputs)
bert_embedding_path='/content/drive/MyDrive/PIL/bert-embedding.jsonl'
bert_run_path='/content/drive/MyDrive/PIL/bert.run'
qrel_path="/content/drive/MyDrive/PIL/test.qrel"
bert_embedding = load_embeddings_from_jsonl(bert_embedding_path)
make_run_from_embeddings(qrel_path,bert_embedding,bert_run_path)
bert_cite_results = qrel_metrics(qrel_path, bert_run_path, metrics=('ndcg', 'map'))
print(bert_cite_results)

#Baseline evaluation untuk ALBERT
albert_tokenizer = AutoTokenizer.from_pretrained('albert-base-v1')
albert_model = AutoModel.from_pretrained('albert-base-v1')
result_path="/content/drive/MyDrive/PIL/albert-embedding.jsonl"
albert_outputs=calculate_embeddings(albert_tokenizer, albert_model, result_path, refs,paper_inputs)
albert_embedding_path='/content/drive/MyDrive/PIL/albert-embedding.jsonl'
albert_run_path='/content/drive/MyDrive/PIL/albert.run'
qrel_path="/content/drive/MyDrive/PIL/test.qrel"
albert_embedding = load_embeddings_from_jsonl(albert_embedding_path)
make_run_from_embeddings(qrel_path,albert_embedding,albert_run_path)
albert_cite_results = qrel_metrics(qrel_path, albert_run_path, metrics=('ndcg', 'map'))
print(albert_cite_results)

#Baseline evaluation untuk SPECTER
specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
specter_model = AutoModel.from_pretrained('allenai/specter')
result_path="/content/drive/MyDrive/PIL/specter-embedding.jsonl"
specter_outputs=calculate_embeddings(specter_tokenizer, specter_model, result_path, refs,paper_inputs)
specter_embedding_path='/content/drive/MyDrive/PIL/specter-embedding.jsonl'
specter_run_path='/content/drive/MyDrive/PIL/specter.run'
qrel_path="/content/drive/MyDrive/PIL/test.qrel"
specter_embedding = load_embeddings_from_jsonl(specter_embedding_path)
make_run_from_embeddings(qrel_path,specter_embedding,specter_run_path)
import pytrec_eval
specter_cite_results = qrel_metrics(qrel_path, specter_run_path, metrics=('ndcg', 'map'))
print(specter_cite_results)

