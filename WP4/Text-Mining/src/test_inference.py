from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, PretrainedConfig
import argparse
import pandas as pd
import os
import glob
import natsort
from peft import LoraConfig, PeftModel
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
        description='Use a fine-tuned model to get predictions for a given dataset')
    parser.add_argument('--test_file_path', type=str, required=True)
    parser.add_argument('--out_file_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--lora', action="store_true")

    args = parser.parse_args()

    test_df = pd.read_table(args.test_file_path, lineterminator="\n")

    model_name = args.model_path.split("/")[-1]
    print(f"Testing {model_name}")
    checkpoints = glob.glob(os.path.join(args.model_path, "checkpoint*"))
    if checkpoints:
        last_checkpoint = natsort.natsorted(checkpoints)[-1]
    else:
        last_checkpoint = args.model_path

    id2label = {int(label_id): label for label_id, label in
                json.load(open(os.path.join(args.model_path, "id2label.json"))).items()}
    label2id = {value: key for key, value in id2label.items()}

    if args.lora:
        config = LoraConfig.from_pretrained(last_checkpoint)

        transformer = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=len(id2label),
            id2label=id2label, label2id=label2id
        )

        transformer = PeftModel.from_pretrained(transformer, last_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"))

    else:
        transformer = AutoModelForSequenceClassification.from_pretrained(last_checkpoint, num_labels=len(id2label),
                                                                         id2label=id2label, label2id=label2id)

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"))

    pipe = pipeline("text-classification", model=transformer, tokenizer=tokenizer, max_length=512,
                    truncation=True)

    x = test_df["tweet_text"].to_list()

    predictions = pipe(x)
    predicted_labels = [pred["label"] for pred in predictions]
    scores = [pred["score"] for pred in predictions]

    results = pd.DataFrame()
    results["tweet_text"] = test_df["tweet_text"]
    results["prediction"] = predicted_labels
    results["prediction_score"] = scores

    results.to_csv(args.out_file_path, index=False, sep="\t", float_format='%.3f')
