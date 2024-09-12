
import argparse
import json
import sys
import time
import pandas as pd
import pickle

from genra_incremental import GenraPipeline


def write_answers(results, topic, emb_model):
    final_answers = []
    q_a = []
    for q, g_answers in results.items():
        for answer in g_answers:
            final_answers.append({'question':q, "tweets":answer['tweets'], "batch":answer['batch_number'], "summary":answer['summary'] })
            for t in answer['tweets']:
                q_a.append((q,t))
    answers_df = pd.DataFrame.from_dict(final_answers)
    pickle.dump(answers_df, open(f'genra_answers_{topic}_{emb_model}_df.pkl', 'wb'))
    q_a = list(set(q_a))
    q_a_df = pd.DataFrame(q_a, columns =['question', 'tweet'])
    q_a_df = q_a_df.sort_values(by=["question"], ascending=False)
    q_a_df.to_csv(f'genra_answers_{topic}_{emb_model}.csv', index=False)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', type=str, default='innsbruck', required=True,
                        help='Name of the topic to process (innsbruck).')
    parser.add_argument('--batch_size', type=int, default=150, required=True,
                        help='Adjust the tweet batch size.')
    parser.add_argument('--llm', type=str, default='mistral', required=True,
                        help='The llm model to use.')
    parser.add_argument('--aggregator', type=str, default="linear", required=True,
                        help='Choose rank aggregation model.')
    parser.add_argument('--topk', type=int, default=5, required=True,
                        help='Number of results to retrieve.')
    #parser.add_argument('summarize_batch', type=bool, default=True, required=True,
    #                    help='Summarize facts of the batch.')

    # embedding model (fixed)
    emb_model = 'multi-qa-mpnet-base-dot-v1'
    #emb_model = 'gtr-t5-xl'
    #emb_model = 'nishimoto/contriever-sentencetransformer'
    
    args = parser.parse_args()

    contexts = [] # in case we have generated context

    # load fixed queries
    queries_df = pd.read_csv('queries_inn.csv')
    queries = queries_df['query'].tolist()

    #load tweet data
    tweets_df = pd.read_csv(f'tweets_{args.topic}.csv')
    tweets = tweets_df['text'].tolist()

    # init Genra model. For now, GenraPipeline creates the needed indexer. Should modify this.
    print('[INFO]: Loading GENRA pipeline....')
    genra = GenraPipeline(args.llm, emb_model, args.aggregator, contexts)
    print('[INFO]: DONE!')
    print('[INFO]: Waiting for data...')

    # slice tweets in batches
    batches = [tweets_df[i:i+args.batch_size] for i in range(0,len(tweets_df),args.batch_size)]
    # for now store answers in a list
    genra_answers = []
    summarize_batch = True
    #try:
    for batch_number, tweets in enumerate(batches):
        # first populate index
        print(f'[INFO]: Populating index for batch {batch_number}')
        genra.qa_indexer.index_dataframe(tweets)
        # now genra retrieval
        print(f'[INFO]: Performing retrieval for batch {batch_number}')
        genra.retrieval(batch_number, queries_df, args.topk, summarize_batch)
        print('[INFO]: Finished!')
    print('[INFO]: Processed all batches!')
    genra_answers = genra.answers_store
    print('[INFO]: Writing answers to file..')
    emb_model = emb_model.replace("/","_")
    write_answers(genra_answers, args.topic, emb_model)
    #print("genra_answers: ", genra_answers)
    #print("questions: ", genra_answers.keys())
    print('[INFO]: Summarizing historical facts...')
    h_summary = genra.summarize_history(queries_df)
    with open(f"genra_final_summary_{args.topic}_{emb_model}.txt", "w") as text_file:
        text_file.write(h_summary)
    #print("h_summary: ", h_summary)
    #except:
    #    print("[ERROR:] Error during processing stream!")


if __name__ == "__main__":
    run()
