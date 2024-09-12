import pandas as pd
from train import TransformerModel, TweetDataset
from json import dump
import os
from datetime import datetime
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

label2id = {"none": 0, "flood": 1, "fire": 2}
id2label = {value: key for key, value in label2id.items()}


def get_num_samples(df):
    counts = df["event"].value_counts()
    keys = label2id.keys()
    samples = dict.fromkeys(keys, 0)
    for key, count in counts.items():
        samples[key] = count

    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
        description='Finetune mBERT to predict event type regarding weather disasters')
    parser.add_argument("-ei", '--experiment_id', type=str)
    parser.add_argument("-r", '--results_folder', type=str, default="results")
    parser.add_argument("-es", '--experiment_suffix', type=str, default="")
    parser.add_argument("-ml", '--max_length', type=int, default=512)
    parser.add_argument("-m", '--model_path', type=str, required=True)
    parser.add_argument('--train_file_path', type=str, required=True)
    parser.add_argument('--few_shot_train_file_path', type=str, default=None)
    parser.add_argument('--validation_file_path', type=str, required=True)
    parser.add_argument('--test_file_path', type=str, required=True)
    parser.add_argument('--fine_tuning', type=str, choices=('normal', 'lora'), default=None)
    parser.add_argument('--multilanguage', action="store_true", help='add this flag to indicate that we will train '
                                                                     'and test with different languages. That way the'
                                                                     'script will show the test results by language')
    args = parser.parse_args()
    now = datetime.now()

    if args.experiment_suffix:
        suffix = "_" + args.experiment_suffix
    else:
        suffix = args.experiment_suffix

    if not args.experiment_id:
        experiment_id = now.strftime("%d%m%Y_%H%M%S") + suffix
    else:
        experiment_id = args.experiment_id + suffix
    experiment_folder = os.path.join(args.results_folder, experiment_id)
    logs_folder = os.path.join(experiment_folder, "logs")

    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(os.path.join(experiment_folder, 'commandline_args.txt'), 'w') as f:
        dump(args.__dict__, f, indent=2)

    train_df = pd.read_table(args.train_file_path, lineterminator="\n")
    train_df.dropna(subset=['event'], inplace=True)
    dev_df = pd.read_table(args.validation_file_path, lineterminator="\n")
    dev_df.dropna(subset=['event'], inplace=True)
    test_df = pd.read_table(args.test_file_path, lineterminator="\n")
    test_df.dropna(subset=['event'], inplace=True)
    print("Read dfs")

    # dataset_samples = {"train": get_num_samples(train_df), "dev": get_num_samples(dev_df),
    #                    "test": get_num_samples(test_df)}

    classes = list(get_num_samples(train_df).keys())

    assert len(classes) <= len(id2label), "The dataset contains more labels than it was originally hardcoded"

    transformer = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3, id2label=id2label,
                                                                     label2id=label2id)
    if not args.fine_tuning:
        for param in transformer.bert.parameters():
            param.requires_grad = False
    print("Loaded pretrained model")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loaded tokenizer")

    if args.fine_tuning:
        if args.fine_tuning == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            transformer = get_peft_model(transformer, peft_config)
            print(transformer.print_trainable_parameters())

    model = TransformerModel(transformer, results_folder=experiment_folder, logs_folder=logs_folder, classes=classes)
    print("Prepared TransformerModel object")

    train_dataset = TweetDataset(train_df, tokenizer, args.max_length, label2id)
    dev_dataset = TweetDataset(dev_df, tokenizer, args.max_length, label2id)
    print("Loaded datasets")
    trainer = model.train_model(train_dataset, dev_dataset)
    print("Finished training")

    if args.few_shot_train_file_path:
        few_shot_train_df = pd.read_table(args.few_shot_train_file_path)
        few_shot_train_dataset = TweetDataset(few_shot_train_df, tokenizer, args.max_length, label2id)
        trainer = model.train_model(few_shot_train_dataset, dev_dataset)
        print("Finished few-shot training")

    # time to compute and report the results
    if args.multilanguage:
        languages = test_df["tweet_language"].unique().tolist()
        results = dict.fromkeys(languages)
        num_samples = {lang: {} for lang in languages}

        for lang in languages:
            lang_test_df = test_df.loc[test_df['tweet_language'] == lang]

            test_dataset = TweetDataset(lang_test_df, tokenizer, args.max_length, label2id)
            results[lang] = model.test_model(test_dataset)

            num_samples[lang]["train"] = get_num_samples(train_df.loc[train_df['tweet_language'] == lang])
            num_samples[lang]["test"] = get_num_samples(lang_test_df)

        # process the results to have a file so it can be copied directly to our google sheets results
        data = []
        for lang, lang_results in results.items():
            lang_train_samples = num_samples[lang]["train"]
            lang_test_samples = num_samples[lang]["test"]
            lang_report = lang_results["test_report"]
            row = [lang, lang_train_samples["fire"], lang_train_samples["flood"], lang_train_samples["none"],
                   lang_test_samples["fire"], lang_test_samples["flood"], lang_test_samples["none"],
                   lang_report["fire"]["precision"], lang_report["flood"]["precision"],
                   lang_report["none"]["precision"],
                   lang_report["fire"]["recall"], lang_report["flood"]["recall"], lang_report["none"]["recall"],
                   lang_report["fire"]["f1-score"], lang_report["flood"]["f1-score"], lang_report["none"]["f1-score"],
                   lang_report["accuracy"]]
            data.append(row)

    else:
        test_dataset = TweetDataset(test_df, tokenizer, args.max_length, label2id)
        results = model.test_model(test_dataset)
        num_samples = dict.fromkeys(["train", "test"])
        num_samples["train"] = get_num_samples(train_df)
        num_samples["test"] = get_num_samples(test_df)

        train_samples = num_samples["train"]

        test_samples = num_samples["test"]
        lang_report = results["test_report"]
        row = [" ", train_samples["fire"], train_samples["flood"], train_samples["none"],
               test_samples["fire"], test_samples["flood"], test_samples["none"],
               lang_report["fire"]["precision"], lang_report["flood"]["precision"], lang_report["none"]["precision"],
               lang_report["fire"]["recall"], lang_report["flood"]["recall"], lang_report["none"]["recall"],
               lang_report["fire"]["f1-score"], lang_report["flood"]["f1-score"], lang_report["none"]["f1-score"],
               results["test_accuracy"]]
        data = [row]

    report = pd.DataFrame(data, columns=["language", "train_num_fire", "train_num_flood", "train_num_none",
                                         "test_num_fire", "test_num_flood", "test_num_none",
                                         "precision_fire", "precision_flood", "precision_none",
                                         "recall_fire", "recall_flood", "recall_none",
                                         "f1_fire", "f1_flood", "f1_none",
                                         "average_accuracy"])

    print("Finished testing")
    # model.precision_recall_curve(test_dataset)
    # print("Finished prc")
    report.to_csv(os.path.join(experiment_folder, "report.tsv"), index=False, sep="\t", float_format='%.3f')
    tokenizer.save_pretrained(os.path.join(experiment_folder, "tokenizer"))
    dump(results, open(os.path.join(experiment_folder, "test_results.json"), "w"), indent=4)
    dump(id2label, open(os.path.join(experiment_folder, "id2label.json"), "w"), indent=4)
