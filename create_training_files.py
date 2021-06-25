
"""

File ini berisi kode untuk memproses traiing triples termasuk tokenisasi menggunakan ALLENNLP:
Panggil menggunakan:
python create_training_files.py --data-dir data --metadata data/metadata.json --outdir data/preprocessed/


"""


import logging
import os

import argparse
import json
import multiprocessing
import pathlib
import pickle
from time import time
from typing import Dict, Optional, Tuple, List, Any

import tqdm
from allennlp.common import Params
from allennlp.data import DatasetReader, TokenIndexer, Token, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter, BertBasicWordSplitter
from allennlp.training.util import datasets_from_params

from multiprocessing import Pool
import multiprocessing

from specter.data_utils import triplet_sampling_parallel



#global variable parameter untuk model BERT
bert_params = {
    "do_lowercase": "true",
    "pretrained_model": "data/scivocab_scivocab_uncased/vocab.txt",
    "use_starting_offsets": "true"
}



# global variables untuk model BERT dan training data
_tokenizer = None
_token_indexers = None
_token_indexer_author_id = None
_token_indexer_author_position = None
_token_indexer_venue = None
_token_indexer_id = None
_max_sequence_length = None
_concat_title_abstract = None
_data_source = None
_included_text_fields = None

MAX_NUM_AUTHORS = 5



def set_values(max_sequence_length: Optional[int] = -1,
               concat_title_abstract: Optional[bool] = None,
               data_source: Optional[str] = None,
               included_text_fields: Optional[str] = None
               ) -> None:
    """ 
    Fungsi untuk menetapkan global values
    input: 
        max_sequence_length: panjang sequence
        concat_title_abstract: apakah melakukan akonkatenasi judul dan abstrak
        data_source : sumber data
        included_text_filed: text field apa yang diproses
    output: none
    """
    
    global _tokenizer
    global _token_indexers
    global _token_indexer_author_id
    global _token_indexer_author_position
    global _token_indexer_venue
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    if _tokenizer is None:  
        _tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter(do_lower_case=bert_params["do_lowercase"]))
        _token_indexers = {"bert": PretrainedBertIndexer.from_params(Params(bert_params))}
        _token_indexer_author_id = {"tokens": SingleIdTokenIndexer(namespace='author')}
        _token_indexer_author_position = {"tokens": SingleIdTokenIndexer(namespace='author_positions')}
        _token_indexer_venue = {"tokens": SingleIdTokenIndexer(namespace='venue')}
        _token_indexer_id = {"tokens": SingleIdTokenIndexer(namespace='id')}
    _max_sequence_length = max_sequence_length
    _concat_title_abstract = concat_title_abstract
    _data_source = data_source
    _included_text_fields = included_text_fields


def get_text_tokens(title_tokens, abstract_tokens, abstract_delimiter):
    """ Fungsi untuk konkatenasi judul dan abstrak menggunakan delimiter
    input: token judul, token abstract, delimiter dari abstract
    output: token gabungan
    """
    if title_tokens[-1] != Token('.'):
            title_tokens += [Token('.')]

    title_tokens = title_tokens + abstract_delimiter + abstract_tokens
    return title_tokens

def get_instance(paper):
    """Fungsi untuk memproses tokenisasi masing2 instance dari paper
    input:
        paper: data paper
    output:
        instance: instance dari paper yang telah tertokenisasi
    """
    
    global _tokenizer
    global _token_indexers
    global _token_indexer_author_id
    global _token_indexer_author_position
    global _token_indexer_venue
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    included_text_fields = set(_included_text_fields.split())

    query_abstract_tokens = _tokenizer.tokenize(paper.get("query_abstract") or "")
    query_title_tokens = _tokenizer.tokenize(paper.get("query_title") or "")

    pos_abstract_tokens = _tokenizer.tokenize(paper.get("pos_abstract") or "")
    pos_title_tokens = _tokenizer.tokenize(paper.get("pos_title") or "")

    neg_abstract_tokens = _tokenizer.tokenize(paper.get("neg_abstract") or "")
    neg_title_tokens = _tokenizer.tokenize(paper.get("neg_title") or "")

    if _concat_title_abstract and 'abstract' in included_text_fields:
        abstract_delimiter = [Token('[SEP]')]
        query_title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
        pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
        neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
        query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []

    if 'authors' in included_text_fields and _max_sequence_length > 0:
        max_seq_len = _max_sequence_length - 15  # reserve max 15 tokens for author names
    else:
        max_seq_len = _max_sequence_length

    if _max_sequence_length > 0:
        query_abstract_tokens = query_abstract_tokens[:max_seq_len]
        query_title_tokens = query_title_tokens[:max_seq_len]
        pos_abstract_tokens = pos_abstract_tokens[:max_seq_len]
        pos_title_tokens = pos_title_tokens[:max_seq_len]
        neg_abstract_tokens = neg_abstract_tokens[:max_seq_len]
        neg_title_tokens = neg_title_tokens[:max_seq_len]

    if 'authors' in included_text_fields:
        source_author_text = ' '.join(paper.get("query_authors") or [])
        pos_author_text = ' '.join(paper.get("pos_authors") or [])
        neg_author_text = ' '.join(paper.get("neg_authors") or [])
        source_author_tokens = _tokenizer.tokenize(source_author_text)
        pos_author_tokens = _tokenizer.tokenize(pos_author_text)
        neg_author_tokens = _tokenizer.tokenize(neg_author_text)

        author_delimiter = [Token('[unused0]')]

        query_title_tokens = query_title_tokens + author_delimiter + source_author_tokens
        pos_title_tokens = pos_title_tokens + author_delimiter + pos_author_tokens
        neg_title_tokens = neg_title_tokens + author_delimiter + neg_author_tokens

    query_venue_tokens = _tokenizer.tokenize(paper.get('query_venue') or NO_VENUE)
    pos_venue_tokens = _tokenizer.tokenize(paper.get('pos_venue') or NO_VENUE)
    neg_venue_tokens = _tokenizer.tokenize(paper.get('neg_venue') or NO_VENUE)

    # pos_year_tokens = _tokenizer.tokenize(paper.get("pos_year"))
    # pos_body_tokens = _tokenizer.tokenize(paper.get("pos_body"))
    #
    # neg_year_tokens = _tokenizer.tokenize(paper.get("neg_year"))
    # neg_body_tokens = _tokenizer.tokenize(paper.get("neg_body"))

    fields = {
        "source_title": TextField(query_title_tokens, token_indexers=_token_indexers),
        "pos_title": TextField(pos_title_tokens, token_indexers=_token_indexers),
        "neg_title": TextField(neg_title_tokens, token_indexers=_token_indexers),
        "source_venue": TextField(query_venue_tokens, token_indexers=_token_indexer_venue),
        "pos_venue": TextField(pos_venue_tokens, token_indexers=_token_indexer_venue),
        "neg_venue": TextField(neg_venue_tokens, token_indexers=_token_indexer_venue),
        'source_paper_id': MetadataField(paper['query_paper_id']),
        "pos_paper_id": MetadataField(paper['pos_paper_id']),
        "neg_paper_id": MetadataField(paper['neg_paper_id']),
    }

    source_authors, source_author_positions = _get_author_field(paper.get("query_authors") or [])
    pos_authors, pos_author_positions = _get_author_field(paper.get("pos_authors") or [])
    neg_authors, neg_author_positions = _get_author_field(paper.get("neg_authors") or [])

    fields['source_authors'] = source_authors
    fields['source_author_positions'] = source_author_positions
    fields['pos_authors'] = pos_authors
    fields['pos_author_positions'] = pos_author_positions
    fields['neg_authors'] = neg_authors
    fields['neg_author_positions'] = neg_author_positions

    if not _concat_title_abstract:
        if query_abstract_tokens:
            fields["source_abstract"] = TextField(query_abstract_tokens, token_indexers=_token_indexers)
        if pos_abstract_tokens:
            fields["pos_abstract"] = TextField(pos_abstract_tokens, token_indexers=_token_indexers)
        if neg_abstract_tokens:
            fields["neg_abstract"] = TextField(neg_abstract_tokens, token_indexers=_token_indexers)

    if _data_source:
        fields["data_source"] = MetadataField(_data_source)

    return Instance(fields)

class TrainingInstanceGenerator:
    #Kelas utama untuk generator training

    def __init__(self,
                 data,
                 metadata,
                 samples_per_query: int = 5,
                 margin_fraction: float = 0.5,
                 ratio_hard_negatives: float = 0.3,
                 data_source: str = None):
        self.samples_per_query = samples_per_query
        self.margin_fraction = margin_fraction
        self.ratio_hard_negatives = ratio_hard_negatives
        self.paper_feature_cache = {}
        self.metadata = metadata
        self.data_source = data_source

        self.data = data
        # self.triplet_generator = TripletGenerator(
        #     paper_ids=list(metadata.keys()),
        #     coviews=data,
        #     margin_fraction=self.margin_fraction,
        #     samples_per_query=self.samples_per_query,
        #     ratio_hard_negatives=self.ratio_hard_negatives
        # )

    def _get_paper_features(self, paper: Optional[dict] = None) -> \
        Tuple[List[Token], List[Token], List[Token], int, List[Token]]:
        """
        Fungsi untuk memproses features dari masing2 paper
        input: 
            paper: dictionary dari paper
        output:
            features: features dari masing2 paper
        """
        if paper:
            paper_id = paper.get('paper_id')
            if paper_id in self.paper_feature_cache:  
                return self.paper_feature_cache[paper_id]

            venue = paper.get('venue') or NO_VENUE
            year = paper.get('year') or 0
            body = paper.get('body')
            authors = paper.get('author-names')
            author_ids = paper.get('authors')
            references = paper.get('references')
            features = paper.get('abstract'), paper.get('title'), venue, year, body, authors, references
            self.paper_feature_cache[paper_id] = features
            return features
        else:
            return None, None, None, None, None, None, None

    def get_raw_instances(self, query_ids, subset_name=None, n_jobs=10):
        """
        Fungsi untuk memproses instance dari paper
        
        input:
            query_ids: list dari query ids dari triplets training
            subset_name: nama opsional untuk subset 
            
        outputs:
            dictionary list dari instances (dictionaries)
        """
        logger.info('Generating triplets ...')
        count_success, count_fail = 0, 0
        # instances = []
        for triplet in triplet_sampling_parallel.generate_triplets(list(self.metadata.keys()), self.data,
                                                            self.margin_fraction, self.samples_per_query,
                                                            self.ratio_hard_negatives, query_ids,
                                                            data_subset=subset_name, n_jobs=n_jobs):
            try:
                query_paper = self.metadata[triplet[0]]
                pos_paper = self.metadata[triplet[1][0]]
                neg_paper = self.metadata[triplet[2][0]]
                count_success += 1

                # check if all papers have title and abstract (all must have title)
                failed = False
                for paper in (query_paper, pos_paper, neg_paper):
                    if not paper['title'] or (not paper['title'] and not paper['abstract']):
                        failed = True
                        break
                if failed:
                    count_fail += 1
                    continue

                query_abstract, query_title, query_venue, query_year, query_body, query_authors, query_refs = \
                    self._get_paper_features(query_paper)
                pos_abstract, pos_title, pos_venue, pos_year, pos_body, pos_authors, pos_refs = self._get_paper_features(pos_paper)
                neg_abstract, neg_title, neg_venue, neg_year, neg_body, neg_authors, neg_refs = self._get_paper_features(neg_paper)

                instance = {
                    "query_abstract": query_abstract,
                    "query_title": query_title,
                    "query_venue": query_venue,
                    "query_year": query_year,
                    "query_body": query_body,
                    "query_authors": query_authors,
                    "query_paper_id": query_paper["paper_id"],
                    "pos_abstract": pos_abstract,
                    "pos_title": pos_title,
                    "pos_venue": pos_venue,
                    "pos_year": pos_year,
                    "pos_body": pos_body,
                    "pos_authors": pos_authors,
                    "pos_paper_id": pos_paper["paper_id"],
                    "neg_abstract": neg_abstract,
                    "neg_title": neg_title,
                    "neg_venue": neg_venue,
                    "neg_year": neg_year,
                    "neg_body": neg_body,
                    "neg_authors": neg_authors,
                    "neg_paper_id": neg_paper["paper_id"],
                    "data_source": self.data_source
                }
                yield instance
            except KeyError:
                # if there is no title and abstract skip this triplet
                count_fail += 1
                pass
        logger.info(f"done getting triplets, success rate:{(count_success*100/(count_success+count_fail+0.001)):.2f}%,"
                     f"total: {count_success+count_fail}")


def get_instances(data, query_ids_file, metadata, data_source=None, n_jobs=1, n_jobs_raw=12,
                  ratio_hard_negatives=0.3, margin_fraction=0.5, samples_per_query=5,
                  concat_title_abstract=False, included_text_fields='title abstract'):
    """
    Fungsi untuk memproses instance allennlp
    
    output:
        List[Instance]
    """

    if n_jobs == 0:
        raise RuntimeError(f"argument `n_jobs`={n_jobs} is invalid, should be >0")

    generator = TrainingInstanceGenerator(data=data, metadata=metadata, data_source=data_source,
                                          margin_fraction=margin_fraction, ratio_hard_negatives=ratio_hard_negatives,
                                          samples_per_query=samples_per_query)

    set_values(max_sequence_length=512,
               concat_title_abstract=concat_title_abstract,
               data_source=data_source,
               included_text_fields=included_text_fields)

    query_ids = [line.strip() for line in open(query_ids_file)]

    instances_raw = [e for e in generator.get_raw_instances(
        query_ids, subset_name=query_ids_file.split('/')[-1][:-4], n_jobs=n_jobs_raw)]

    if n_jobs == 1:
        logger.info(f'converting raw instances to allennlp instances:')
        for e in tqdm.tqdm(instances_raw):
            yield get_instance(e)

    else:
        logger.info(f'converting raw instances to allennlp instances ({n_jobs} parallel jobs)')
        with Pool(n_jobs) as p:
            instances = list(tqdm.tqdm(p.imap(
                get_instance, instances_raw)))

        # multiprocessing does not work as generator, needs to generate everything
        # see: https://stackoverflow.com/questions/5318936/python-multiprocessing-pool-lazy-iteration
        return instances


def main(data_files, train_ids, val_ids, test_ids, metadata_file, outdir, n_jobs=1, njobs_raw=1,
         margin_fraction=0.5, ratio_hard_negatives=0.3, samples_per_query=5, comment='', bert_vocab='',
         concat_title_abstract=False, included_text_fields='title abstract'):
    """
    Fungsi untuk memproses isntances dari list datafiles kemudian disimpan kedalam file
    inputs:
        data_files: list file
        train_ids: list training paper ids 
        val_ids: list validation paper ids
        test_ids: list  test paper ids
        metadata_file: path file metadata 
        outdir: path  output directory
        n_jobs: jumlah  parallel jobs untuk konversi instances
        njobs_raw: jumlah  parallel jobs untuk generasi triplets
        margin_fraction: parameter untuk triplet generation
        ratio_hard_negatives: jumlah dari hard negatives
        samples_per_query: jumlah samples per query paper
        comment: custom comment 


    Returns:
        Nothing (menyimpan langsung ke file sesuai dengan masing2 data file)
    """
    global bert_params
    bert_params['pretrained_model'] = bert_vocab

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(metadata_file) as f_in:
        logger.info(f'loading metadata: {metadata_file}')
        metadata = json.load(f_in)

    for data_file, train_set, val_set, test_set in zip(data_files, train_ids, val_ids, test_ids):
        logger.info(f'loading data file: {data_file}')
        with open(data_file) as f_in:
            data = json.load(f_in)
        data_source = data_file.split('/')[-1][:-5]  # e.g., coviews_v2012
        if comment:
            data_source += f'-{comment}'

        metrics = {}
        for ds_name, ds in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
            logger.info(f'getting instances for `{data_source}` and `{ds_name}` set')
            outfile = f'{outdir}/{data_source}-{ds_name}.p'
            logger.info(f'writing output {outfile}')
            with open(outfile, 'wb') as f_in:
                pickler = pickle.Pickler(f_in)
                # pickler.fast = True
                idx = 0
                for instance in get_instances(data=data,
                                              query_ids_file=ds,
                                              metadata=metadata,
                                              data_source=data_source,
                                              n_jobs=n_jobs, n_jobs_raw=njobs_raw,
                                              margin_fraction=margin_fraction,
                                              ratio_hard_negatives=ratio_hard_negatives,
                                              samples_per_query=samples_per_query,
                                              concat_title_abstract=concat_title_abstract,
                                              included_text_fields=included_text_fields):
                    pickler.dump(instance)
                    idx += 1
                    # to prevent from memory blow
                    if idx % 2000 == 0:
                        pickler.clear_memo()
            metrics[ds_name] = idx
        with open(f'{outdir}/{data_source}-metrics.json', 'w') as f_out2:
            json.dump(metrics, f_out2, indent=2)



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', help='path merujuk pada directory yang mengandung `data.json`, `train.csv`, `dev.csv` and `test.csv` files')
    ap.add_argument('--metadata', help='path file metadata ')
    ap.add_argument('--outdir', help='output directory ')
    ap.add_argument('--njobs', help='jumlah parallel jobs untuk konversi instance ', default=1, type=int)
    ap.add_argument('--njobs_raw', help='jumlah  parallel jobs untuk generasi triplet ', default=12, type=int)
    ap.add_argument('--ratio_hard_negatives', default=0.3, type=float)
    ap.add_argument('--samples_per_query', default=5, type=int)
    ap.add_argument('--margin_fraction', default=0.5, type=float)
    ap.add_argument('--comment', default='', type=str)
    ap.add_argument('--data_files', help='space delimted list dari data files', default=None)
    ap.add_argument('--bert_vocab', help='path dari bert vocab', default='data/scibert_scivocab_uncased/vocab.txt')
    ap.add_argument('--concat-title-abstract', action='store_true', default=False)
    ap.add_argument('--included-text-fields', default='title abstract', help=' delimieted list dari fields `')
    args = ap.parse_args()

    data_file = os.path.join(args.data_dir, 'data.json')
    train_ids = os.path.join(args.data_dir, 'train.txt')
    val_ids = os.path.join(args.data_dir, 'val.txt')
    test_ids = os.path.join(args.data_dir, 'test.txt')

    if args.metadata:
        metadata_file = args.metadata

    init_logger()
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

    main([data_file], [train_ids], [val_ids], [test_ids], metadata_file, args.outdir, args.njobs, args.njobs_raw,
         margin_fraction=args.margin_fraction, ratio_hard_negatives=args.ratio_hard_negatives,
         samples_per_query=args.samples_per_query, comment=args.comment, bert_vocab=args.bert_vocab,
         concat_title_abstract=args.concat_title_abstract, included_text_fields=args.included_text_fields
         )
