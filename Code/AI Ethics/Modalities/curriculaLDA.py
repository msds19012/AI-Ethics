import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from matplotlib import pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'www', 'com', 'http', 'fall', 'spring', 'https', 'objective', 'syllabus', 'weekly', 'student', 'pm', 'class', 'due', 'assignment', 'grade', 'lecture' 'week',  'year', 'semester', 'grade', 'question', 'week', 'link', 'read', 'instructor'])

# pdminer
from pdfminer.high_level import extract_text
import glob

# beautifulSoup4 and docx2txt
from bs4 import BeautifulSoup
from bs4.element import Comment
from os import listdir
import docx2txt

# tqdm
from tqdm import tqdm

# Function to get visible text from html
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

# Load data
def loadCurricula():
    global_pdf_text = dict()
    def extractText(path):
        pdf_path = path
        if path in global_pdf_text:
            return global_pdf_text[path]
        else:
            text = extract_text(pdf_path)
            global_pdf_text[path] = text
            return text

    courses = glob.glob("courses/pdf/*")
    print(len(courses))
    courses_dict = dict()
    for index, pdf in tqdm(enumerate(courses), total=len(courses)):
        text = extractText(pdf)
        courses_dict[courses[index].split('/')[-1]] = text

    courses = glob.glob("courses/docx/*")
    print(len(courses))
    for index, docx in tqdm(enumerate(courses), total=len(courses)):
        text = docx2txt.process(docx)
        courses_dict[courses[index].split('/')[-1]] = text

    courses = glob.glob("courses/html/*")
    print(len(courses))
    for index, html in tqdm(enumerate(courses), total=len(courses)):
        html_file=open(html, 'r')
        try:
            text = text_from_html(html_file.read())
            courses_dict[courses[index].split('/')[-1]] = text
        except:
            print("Error processing", courses[index].split('/')[-1])

    courses = glob.glob("courses/txt/*")
    print(len(courses))
    for index, txt in tqdm(enumerate(courses), total=len(courses)):
        text = open(txt, "r").read()
        courses_dict[courses[index].split('/')[-1]] = text

    df_courses = pd.DataFrame(courses_dict.items(), columns=['filename', 'text'])
    return df_courses

def loadSDGPapers():
    global_pdf_text = dict()
    def extractText(path):
        pdf_path = path
        if path in global_pdf_text:
            return global_pdf_text[path]
        else:
            text = extract_text(pdf_path)
            global_pdf_text[path] = text
            return text

    courses = glob.glob("../SDGS-Papers/*.pdf")
    courses_dict = dict()
    for index, pdf in tqdm(enumerate(courses), total=len(courses)):
        text = extractText(pdf)
        courses_dict[courses[index].split('/')[-1]] = text

    df_courses = pd.DataFrame(courses_dict.items(), columns=['filename', 'text'])
    return df_courses

def loadData():
    global_pdf_text = dict()
    def extractText(path):
        pdf_path = path
        if path in global_pdf_text:
            return global_pdf_text[path]
        else:
            text = extract_text(pdf_path)
            global_pdf_text[path] = text
            return text

    enablers_papers = glob.glob("SDG-AI-Impact/e-*.pdf")
    inhibitor_papers = glob.glob("SDG-AI-Impact/i-*.pdf")
    courses = glob.glob("courses/*.pdf")

    enabler_text = []
    for pdf in enablers_papers:
        text = extractText(pdf)
        enabler_text.append(text)

    inhibitor_text = []
    for pdf in inhibitor_papers:
        text = extractText(pdf)
        inhibitor_text.append(text)

    courses_text = []
    for pdf in courses:
        text = extractText(pdf)
        courses_text.append(text)
    
    df_enabler = pd.DataFrame(enabler_text)
    df_inhibitor = pd.DataFrame(inhibitor_text)
    df_courses = pd.DataFrame(courses_text)

    df = pd.concat([df_enabler, df_inhibitor, df_courses])
    
    return df

# Data Preprocessing
def preprocessData(df):

    # Remove emails and newline characters
    # Convert to list
    data = df["text"].values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    # Creating Bigram and Trigram Models
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Creating Bigram and Trigram Models

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    import subprocess
    print(subprocess.getoutput("python -m spacy download en"))
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    dictionary = corpora.Dictionary(data_lemmatized)
    dictionary.filter_extremes(no_below=10, no_above=0.6)
    id2word = dictionary

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word, data_lemmatized

#loadCurricula()
