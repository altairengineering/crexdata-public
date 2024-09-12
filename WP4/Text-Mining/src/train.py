from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy, logging
import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import os

logging.set_verbosity_error()


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, label_enc):
        texts = df["tweet_text"].to_list()

        y = df["event"].apply(lambda x: label_enc[x]).to_list()
        x = tokenizer(texts, padding=True, max_length=max_length, truncation=True)

        self.encodings = x
        self.labels = y

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TransformerModel:

    def __init__(self, model, classes, max_length=50, results_folder="results", logs_folder="logs"):
        self.results_folder = results_folder
        self.logs_folder = logs_folder
        self.max_length = max_length
        self.model = model
        self.classes = classes
        self.trainer = None

    def train_model(self, train_dataset, dev_dataset):
        eval_steps = int(len(train_dataset) / 20)
        # Define training args
        training_args = TrainingArguments(
            output_dir=self.results_folder,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            # PyTorch 2.0 specifics
            # bf16=True,  # bfloat16 training
            # torch_compile=True,  # optimizations
            # optim="adamw_torch_fused",  # improved optimizer
            # logging & evaluation strategies
            logging_dir=self.logs_folder,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=eval_steps,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            disable_tqdm=False
            # push to hub parameters
            # report_to="tensorboard",
            # hub_strategy="every_save"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

    def test_model(self, test_dataset):
        return self.trainer.evaluate(test_dataset, metric_key_prefix="test")

    def precision_recall_curve(self, test_dataset):
        result = self.trainer.predict(test_dataset)
        predictions = softmax(torch.tensor(result.predictions), dim=1)
        diff = np.array(list(map(lambda x: (x[1] - x[0]).item(), predictions)))
        binary_prob = (diff + 1) / 2

        precision, recall, thresholds = precision_recall_curve(result.label_ids, binary_prob, pos_label=1)
        f1_scores = 2 * recall * precision / (recall + precision)

        curve = pd.DataFrame(
            {"precision": precision[:-1], "recall": recall[:-1], "f1": f1_scores[:-1], "threshold": thresholds})
        curve.to_csv(os.path.join(self.results_folder, 'precision_recall.tsv'), index=False, sep="\t")

        best_th_ix = np.nanargmax(precision[:-1])
        best_thresh = thresholds[best_th_ix]
        average_precision = average_precision_score(result.label_ids, binary_prob, pos_label=1)
        display = PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_name="mBERT",
            pos_label=1)
        display.plot(name="Precision-Recall curve")
        display.ax_.set_title("Test Data")
        display.ax_.plot(recall[best_th_ix], precision[best_th_ix], "ro",
                         label=f"precisionmax (th = {best_thresh:.2f})")
        display.ax_.legend()
        plt.savefig(os.path.join(self.results_folder, 'precision_recall.png'))

    def compute_metrics(self, predictions):
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        logits, labels = predictions
        predictions = np.argmax(logits, axis=-1)

        metrics = {}
        metrics.update(accuracy_metric.compute(predictions=predictions, references=labels))
        metrics.update(f1_metric.compute(predictions=predictions, references=labels, average="macro"))

        metrics["report"] = classification_report(labels, predictions, labels=np.arange(0, len(self.classes)),
                                                  target_names=self.classes,
                                                  digits=4, output_dict=True)

        return metrics
