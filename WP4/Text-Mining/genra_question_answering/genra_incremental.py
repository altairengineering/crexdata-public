import torch
import time
import numpy as np
import pandas as pd
import pickle
import json
import gzip
import ir_datasets
import re
import os
import requests
# Import the PyFLAGR modules for rank aggregation
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian
import pyflagr.MarkovChains as MarkovChains
import pyflagr.Kemeny as Kemeny
import pyflagr.RRA as RRA
import pyflagr.Weighted as Weighted

from operator import itemgetter

from haystack import Document
from haystack.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from sentence_transformers import SentenceTransformer

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline


class LLMGenerator:
    def __init__(self, llm_model, tokenizer, llm_name):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.llm_name = llm_name

    def generate_answer(self, texts, query, mode='validate'):
        template_texts =""
        for i, text in enumerate(texts):
            template_texts += f'{i+1}. {text} \n'

        if mode == 'validate':
            #conversation = [ {'role': 'user', 'content': f'For the following query and document, judge whether the document can truly answer the query. Output strictly “Yes” or “No” and do NOT explain your answer.\nQuery: {query}\nDocument: {template_texts}'} ]
            conversation = [ {'role': 'user', 'content': f'Given the following query: "{query}"? \nIs the following document relevant to answer this query?\n{template_texts} \nResponse: Yes / No'} ]
        elif mode == 'summarize':
            conversation = [ {'role': 'user', 'content': f'For the following query and documents, try to answer the given query based on the documents.\nQuery: {query} \nDocuments: {template_texts}.'} ]
        elif mode == 'h_summarize':
            #conversation = [ {'role': 'user', 'content': f'Write a brief summary for the following documents in the form of a paragraph highlighting the most crucial information. \nDocuments: {template_texts}'} ]
            conversation = [ {'role': 'user', 'content': f'The documents below describe a developing disaster event. Based on these documents, write a brief summary in the form of a paragraph, highlighting the most crucial information. \nDocuments: {template_texts}'} ]

        #print(conversation)
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        #print(self.llm_model)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device) 
        #print(inputs)
        #print(self.llm_model.device)
        outputs = self.llm_model.generate(**inputs, use_cache=True, max_length=4096,do_sample=True,temperature=0.7,top_p=0.95,top_k=10,repetition_penalty=1.1)
        output_text = self.tokenizer.decode(outputs[0]) 
        if self.llm_name == "solar":
            assistant_respond = output_text.split("Assistant:")[1]
        else:
            assistant_respond = output_text.split("[/INST]")[1]
        #print('respond: ', output_text)
        if mode == 'validate':
            if 'Yes' in assistant_respond:
                return True
            else:
                return False
        elif mode == 'summarize':
            return assistant_respond
        elif mode == 'h_summarize':
            #print('respond: ', output_text)
            return assistant_respond


class QAIndexer:
    def __init__(self, index_type, emb_model):
        self.document_embedder = SentenceTransformersDocumentEmbedder(model=emb_model)
        self.document_embedder.warm_up()
        if index_type == 'in_memory':
            self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")


    def index(self, docs_to_index):
        documents_with_embeddings = self.document_embedder.run(docs_to_index)
        self.document_store.write_documents(documents_with_embeddings['documents'])

    
    def index_dataframe(self, data):
        docs_to_index = self.read_dataframe(data)
        self.index(docs_to_index)
    
    def index_stream(self, stream_data):
        docs_to_index = self.read_stream(stream_data)
        self.index(docs_to_index)
    
    def read_dataframe(self, data):
        # Convert Dataframe to list of dicts for DocumentStore
        docs_to_index = [Document(content=row['text'],id=str(row['order'])) for idx, row in data.iterrows()]
        return docs_to_index
    
    def read_stream(self, stream_data):
        # stream consist of single docs for now
        #print(stream_data)
        docs_to_index = [Document(content=doc['text'],id=str(doc['id'])) for doc in [stream_data]]
        #print(docs_to_index)
        return docs_to_index


class QARetriever:
    def __init__(self, document_store):
        self.retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    def retrieve(self, query, topk):
        retrieval_results = self.retriever.run(query_embedding=query, top_k=topk)
        documents = [x.to_dict() for x in retrieval_results["documents"]]
        return documents


def rank_aggregation(aggregator, lists, k):
    if aggregator == 'linear':
        csum = Linear.CombSUM(norm='score')
        df_out, df_eval = csum.aggregate(input_file=lists)
    elif aggregator == 'outrank':
        outrank = Majoritarian.OutrankingApproach(eval_pts=7)
        df_out, df_eval = outrank.aggregate(input_file=lists)

    df_out['query_ids'] = df_out.index
    queries = list(df_out['query_ids'].unique())
    results = []
    for query in queries:
        df_query = df_out[df_out['query_ids'] == query][:k]
        rank = 0
        for index, r_q in df_query.iterrows():
            rank += 1
            doc_id = r_q['Voter']
            score_doc = r_q['Score']
            results.append({'qid': query, 'docid': doc_id, 'rank': rank, 'score': score_doc})
    return results


class GenraPipeline:
    def __init__(self, llm_name, emb_model, aggregator, contexts):
        self.qa_indexer = QAIndexer('in_memory', emb_model)
        self.qa_retriever = QARetriever(self.qa_indexer.document_store)
        self.encoder = SentenceTransformer(emb_model)
        self.contexts = contexts
        self.aggregator = aggregator
        self.answers_store = {}
        if llm_name == 'solar':
            self.tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0", use_fast=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
            "Upstage/SOLAR-10.7B-Instruct-v1.0",
            device_map="cuda",
            torch_dtype=torch.float16,
            )
            self.llm_generator = LLMGenerator(self.llm_model, self.tokenizer, llm_name)
        elif llm_name == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device_map="cuda",
            torch_dtype=torch.float16,
            )
            self.llm_generator = LLMGenerator(self.llm_model, self.tokenizer, llm_name)
    
    def retrieval(self, batch_number, queries, topk, summarize_results=True):
        #retrieval_results = []
        for qid,question in tqdm(queries[['id','query']].values):
            if len(self.contexts)<1:
                self.contexts.append(question)
            all_emb_c = []
            for c in self.contexts:
                c_emb = self.encoder.encode([c], convert_to_numpy=True)[0]
                #print(c_emb)
                all_emb_c.append(np.array(c_emb))
            all_emb_c = np.array(all_emb_c)
            avg_emb_c = np.mean(all_emb_c, axis=0)
            avg_emb_c = avg_emb_c.reshape((1, len(avg_emb_c)))
            # we want a list of floats for haystack retrievers
            #print(avg_emb_c.shape)
            hits = self.qa_retriever.retrieve(avg_emb_c[0].tolist(), 20) # topk or more?
            hyde_texts = []
            candidate_texts = []
            hit_count = 0
            while len(candidate_texts) < 5:
                if hit_count < len(hits):
                    json_doc = hits[hit_count]
                    #print(json_doc)
                    doc_text = json_doc['content'] 
                    if self.llm_generator.generate_answer([doc_text], question, mode='validate'):
                        candidate_texts.append(doc_text) #candidate_texts.append(doc_text[0])
                    hit_count += 1
                else:
                    break
            if len(candidate_texts)<1:
                # no unswerable result
                #print('[Info]: Cannot find answerable content!')
                results = []
            else:
                all_emb_c = []
                all_hits = []
                for i, c in enumerate(candidate_texts):
                    c_emb = self.encoder.encode([c], convert_to_numpy=True)[0]
                    #prnt(c_emb)
                    c_emb = c_emb.reshape((1, len(c_emb)))
                    c_hits = self.qa_retriever.retrieve(c_emb[0].tolist(), topk) # changed to len(candidates)+1
                    rank=0
                    for hit in c_hits: # get each ranking with pyflagr format
                        rank += 1
                        # penalize score wrt hit counts (the smaller the better!)
                        #all_hits.append({'qid': qid, 'voter':i, 'docid': hit.docid, 'rank': rank, 'score': hit.score/float(hit_count)})
                        all_hits.append({'qid': qid, 'voter':i, 'docid': hit['id'], 'rank': rank, 'score': hit['score']})
                # write pyglagr aggregation files
                tempfile = 'temp_rankings_file'
                with open(tempfile, 'w') as f:
                    for res in all_hits:
                        f.write(f"{res['qid']},V{res['voter']},{res['docid']},{res['score']},test\n")
                # run aggregation
                results = rank_aggregation(self.aggregator, tempfile, topk)
            
                # enhance each result with doc info
                for res in results:
                    #print(res['docid'])
                    #print(self.qa_indexer.document_store.count_documents())
                    #print('filt ', self.qa_indexer.document_store.filter_documents(filters={'id':str(res['docid'])}))
                    res['document'] = self.qa_indexer.document_store.filter_documents(filters={'id':str(res['docid'])})[0].content

            if summarize_results:
                summary = self.summarize_results(question, results, candidate_texts)

            self.store_results(batch_number, question, results, summary)
            #print('ANSWERS..:#################')
            #print(self.answers_store)
            #retrieval_results.append((question, summary))


    def store_results(self, batch_number, question, results, summary):
        if results:
            tweets = [t['document'] for t in results]
            if question in self.answers_store:
                self.answers_store[question].append({'batch_number':batch_number, 'tweets':tweets, 'summary':summary})
            else:
                self.answers_store[question] = [{'batch_number':batch_number, 'tweets':tweets, 'summary':summary}]
                


    def summarize_results(self, question, results, candidate_texts):
        if results:
            #print("candidate_texts ", candidate_texts)
            texts = [t['document'] for t in results] #+ candidate_texts
            #print("texts", texts)
            summary = self.llm_generator.generate_answer(texts, question, mode='summarize')
        else:
            summary = "N/A"
        return summary

    def summarize_history(self, queries):
        h_per_q = []
        for qid,question in tqdm(queries[['id','query']].values):
            if question in self.answers_store:
                q_history = self.answers_store[question]
                q_hist_docs = self.order_history(q_history)
                h_per_q.extend(q_hist_docs)
        #all_sums = [h_per_q]
        #print("summaries: ", h_per_q)
        historical_summary = self.llm_generator.generate_answer(h_per_q, question, mode='h_summarize')
        return historical_summary
    
    def order_history(self, query_history):
        ordered_history = sorted(query_history, key=itemgetter('batch_number'))
        ordered_docs = [hist['summary'] for hist in ordered_history]
        #print("ordered_docs: ", ordered_docs)
        return ordered_docs


