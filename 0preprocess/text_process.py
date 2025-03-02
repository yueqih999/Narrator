import regex as re
import os
import nltk
import pandas as pd
import string
import unicodedata
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from collections import Counter
from math import log

nltk.download('punkt_tab', download_dir='data/nltk_data')
nltk.download('stopwords', download_dir='data/nltk_data')

nltk.data.path.append('data/nltk_data')

def preprocess_txt_to_chapter_csvs(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    text = None

    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                text = f.read()
                print(f"Decoding using {encoding} succeed")
                break
        except UnicodeDecodeError:
            print(f"Try {encoding} failed...")
        
    chapter_pattern = r'(Chapter \d+|第\s*\d+\s*章)'
    chapters = re.split(chapter_pattern, text)
    
    chapters = [ch.strip() for ch in chapters if ch.strip()]
    
    if len(chapters) <= 1:
        chapters = [text]
    
    csv_files = []

    for i, chapter in enumerate(chapters):
        if re.match(chapter_pattern, chapter):
            continue
 
        sentences = sent_tokenize(chapter)
        
        df = pd.DataFrame({
            'sentence_id': range(len(sentences)),
            'sentence': sentences
        })
        
        output_file = os.path.join(output_dir, f'chapter_{i+1}.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        csv_files.append(output_file)
        
        print(f"Generate chapter {i+1} CSV, with {len(sentences)} sentences")
    
    return csv_files


def clean_sentence(sentence, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    sentence = sentence.lower()
    sentence = re.sub(r'\p{P}+', '', sentence)

    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = ''.join([c for c in sentence if not unicodedata.combining(c)])
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    try:
        tokens = word_tokenize(sentence)
    except LookupError:
        print("NLTK word_tokenize failed, using spaces to split")
        tokens = sentence.split()
    
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return sentence, filtered_tokens

def process_csv_with_nlp(csv_file, output_file=None, language='english'):
    stop_words = set(stopwords.words(language))
    df = pd.read_csv(csv_file)
    
    df['cleaned_sentence'] = ''
    df['tokens'] = ''
    
    for i, row in df.iterrows():
        cleaned, tokens = clean_sentence(row['sentence'], stop_words)
        df.at[i, 'cleaned_sentence'] = cleaned
        df.at[i, 'tokens'] = ' '.join(tokens)
    
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Generating CSV file: {output_file}")
    
    return df

def calculate_term_frequency(csv_files, output_file=None):
    all_tokens = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        for tokens_str in df['tokens']:
            if isinstance(tokens_str, str):
                all_tokens.extend(tokens_str.split())
    
    word_count = Counter(all_tokens)
    
    tf_df = pd.DataFrame({
        'word': list(word_count.keys()),
        'frequency': list(word_count.values())
    })
    
    tf_df = tf_df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    if output_file:
        tf_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Generating term frequency file: {output_file}")
    
    return tf_df

def calculate_mutual_information(csv_files, output_file=None, window_size=5, min_freq=10):
    all_tokens = []

    for file in csv_files:
        df = pd.read_csv(file)
        for tokens_str in df['tokens']:
            if isinstance(tokens_str, str):
                all_tokens.extend(tokens_str.split())

    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens, window_size=window_size)
    
    finder.apply_freq_filter(min_freq)
    
    bigram_mi = finder.score_ngrams(bigram_measures.pmi)
    
    mi_df = pd.DataFrame({
        'word1': [bigram[0] for bigram, score in bigram_mi],
        'word2': [bigram[1] for bigram, score in bigram_mi],
        'MI_score': [score for bigram, score in bigram_mi]
    })
  
    mi_df = mi_df.sort_values('MI_score', ascending=False).reset_index(drop=True)
    
    if output_file:
        mi_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"MI file generated: {output_file}")
    
    return mi_df


def main(input_txt, output_base_dir):
    csv_dir = os.path.join(output_base_dir, 'chapter_csvs')
    processed_dir = os.path.join(output_base_dir, 'processed_csvs')
    analysis_dir = os.path.join(output_base_dir, 'analysis')
    
    for dir_path in [csv_dir, processed_dir, analysis_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print("Step 1: Processing text to csv files...")
    csv_files = preprocess_txt_to_chapter_csvs(input_txt, csv_dir)

    print("\nStep 2: clean text...")
    processed_files = []
    for csv_file in csv_files:
        chapter_name = os.path.basename(csv_file)
        output_file = os.path.join(processed_dir, f"processed_{chapter_name}")
        process_csv_with_nlp(csv_file, output_file)
        processed_files.append(output_file)
    
    print("\nStep 3: calculating term frequency...")
    tf_output = os.path.join(analysis_dir, 'term_frequency.csv')
    tf_df = calculate_term_frequency(processed_files, tf_output)
    
    print("\nStep 4: calculating MI values...")
    mi_output = os.path.join(analysis_dir, 'mutual_information.csv')
    mi_df = calculate_mutual_information(processed_files, mi_output)
    
    print(f"\nFinished! All output files saved at: {output_base_dir}")
    print("\nTop 10 term frequency:")
    print(tf_df.head(10))
    
    print("\nTop 10 MI values:")
    print(mi_df.head(10))

if __name__ == "__main__":
    input_txt = "data/01 - The Fellowship Of The Ring.txt"
    output_dir = "data"
    
    main(input_txt, output_dir)



