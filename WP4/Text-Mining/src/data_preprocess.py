import re
import os
import argparse

import pandas as pd

from tqdm import tqdm
from lingua import LanguageDetectorBuilder, Language


REGEX_DICT = {
    "REG_EMAIL" : (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", "<EMAIL>"),
    "REG_UNAME" : (r"@[^\s]+", "<USERNAME>"),
}

def process_tweet(tweet):
    tweet = re.sub(r'@[^\s]+', '', tweet)
    tweet = re.sub(r'http[^\s]+', '', tweet)
    tweet = re.sub(r'#[^\s]+', '', tweet)
    tweet = re.sub(r'&\w+;', '', tweet)
    return tweet

def try_detect(text, detector):
    lang = detector.detect_language_of(process_tweet(text))

    if lang:
        return lang.iso_code_639_1.name.lower()

    return "unknown"

def anonymize(text):
    for _,v in REGEX_DICT.items():
        text = re.sub(v[0], v[1], text)
    return text

def lang_detect(data, textcol):
    ## Languages to detect 
    languages_to_detect = [Language.ENGLISH, Language.GERMAN, Language.SPANISH, Language.CATALAN, Language.DUTCH]
    detector = LanguageDetectorBuilder.from_languages(*languages_to_detect).with_minimum_relative_distance(0.9).build()
    tqdm.pandas(desc="detecting language...")
    data["tweet_language"] = data.progress_apply(lambda x: try_detect(x[textcol], detector), axis=1)
    return data

def anonymizer(data, textcol):
    tqdm.pandas(desc="annonymizing...")
    data[textcol] = data[textcol].progress_apply(anonymize)
    return data
    
def main():
    parser = argparse.ArgumentParser(
        description='Preprocess tsv datasets for training or inference.\n Detects language and annoymizes text.\n Supported languages: English, Spanish, Catalan, German, Dutch')
    parser.add_argument('-f', '--file-path', dest='file_path', type=str, required=True, help='Path of tsv file to preprocess')
    parser.add_argument('--text-colname', dest='text_colname', type=str, required=True, help='tweet or text column name in tsv file')
    parser.add_argument('-o', '--output-path', dest='out_path', type=str, required=True, help='Path to save tsv')
    args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    textcol = args.text_colname    
    data = pd.read_table(args.file_path)
    data = data.dropna(subset=[textcol])
    
    final_df = data.pipe(lang_detect, textcol=textcol).pipe(anonymizer, textcol=textcol)
    
    final_df.rename(columns={textcol: "tweet_text"}, inplace=True)
    file_name = os.path.basename(args.file_path).split(".")[0]
    file_name += "_processed.tsv"
    final_df.to_csv(os.path.join(args.out_path, file_name), index=False, sep="\t", lineterminator="\n")

if __name__ == '__main__':
    main()


