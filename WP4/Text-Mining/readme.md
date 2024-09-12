Crexdata Task 4.5
============

This repository contains a couple of files demonstrating the basic functionality of our work so far in Task 4.5.
The aim of this task is to monitor a social media platform in order to detect in real time posts that
contain relevant information about the ongoing weather incident of interest.

The first step then is to assess whether a post contains relevant information. To do this, we fine-tune a BERT-based
model using datasets tagged for this purpose, that is, each document is tagged whether they contain information about
an event or not. Right now, the classes are Fire (ie. containing information regarding a fire), Flood and None (no
relevant
information or something completely unrelated).

The second step is to aggregate all the posts marked as relevant and query a Question-Answering model with the
mentioned tweets as context, so we can summarize all information or ask questions regarding the content of these
documents.

Scripts
------------
There are three (may increase) main scripts in this repository, main_train.py (with train.py as auxiliary),
test_inference.py and data_process.py.

### Training an event prediction model ###

To train an event prediction model, we mainly need a dataset to train from, a .tsv file with at least the columns
"event" (fire, flood, none, etc.) and "tweet_text". It should also include a column "tweet_language" if the multilanguage
parameter wants to be used.

#### Usage ####

Parameters:

    -ei/--experiment_id: name of the experiment (default: current date-time)

    -r/--results_folder: folder in which to save the output files (default: "results")

    -es/--experiment_suffix: any other information to add to the experiment id (optional)

    -ml/--max_length: maximum number of tokens per input sentence (default: 512)

    -m/--model_path: path to the base model or complete name to be downloaded from huggingface hub (models we have tested:
    bert-base-multilingual-cased, distilbert-base-multilingual-cased, cardiffnlp/twitter-xlm-roberta-base,
    Twitter/twhin-bert-base)

    --train_file_path: path to training file

    --few_shot_train_file_path: path to few-shot training file (optional)

    --validation_file_path: path to validation file

    --test_file_path: path to test file

    --fine_tuning: which technique to use to fine-tune the model, choices=('normal', 'lora'), (default: no fine-tuning,
    NOT recommended)

    --multilanguage: mainly for testing, the final results will be displayed by language

Example:

    python src/main_train.py -m Twitter/twhin-bert-base 
    --train_file_path data/train_file.tsv 
    --validation_file_path data/val_file.tsv 
    --test_file_path data/test_file.tsv --fine_tuning normal

#### Resulting files ####

By default, all the output files will be placed in the "results" folder. Inside, each experiment will have a folder
named after the experiment_id+experiment_suffix, which will in turn contain the checkpoints, the configuration, the tokenizer
and the test results (report.tsv and test_results.json contain roughly the same information but in different format).

### Testing an event prediction model ###

Once the user has obtained a fine-tuned model in some way (either training it or acquiring it from somewhere else), it can be
tested with any other datasets available using test_inference.py.

#### Usage ####

Parameters:

    --test_file_path: path to test file

    --out_file_path: path to output file

    --model_path: path to fine-tuned model. The folder should contain a "tokenizer" folder, an "id2label.json" file, 
    and the checkpoints, which can be organized in folders (checkpoint-5, checkpoint-10, this is the way the training 
    script saves the checkpoints) or we can just place "model.safetensors" and "config.json" along with the rest of the 
    mentioned files.

    --lora: if we have used LoRA to fine-tune a model we need to specify it because the loading is a bit different

#### Resulting files ####

out_file_path will contain three columns: "tweet_text" (the original text), "prediction" (the predicted class), 
"prediction_score" (the score for the predicted class).

### Preprocessing data ##

data_preprocess.py:

Description

Processes a TSV file to detect language in social media post and anonymize them by replacing usernames (@)
with "<USERNAME>", and emails with "<EMAIL>". Detected languages are saved in column "tweet_language" and column with
social media posts is rename to "tweet_text". The dataset is then exported as TSV at output path as
FILENAME_processed.tsv

Usage:

    python data_preprocess.py -f /INPUT_PATH/FILENAME.tsv -o /OUTPUT_PATH/ --text-colname POST_COLUMN_NAME
