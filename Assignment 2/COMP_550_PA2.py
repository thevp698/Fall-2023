'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
                    #print(instances[my_id])
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def convert_synset(sense_key):
    sense_synset = wn.lemma_from_key(sense_key[0])
    return sense_synset.synset().name()

def to_ascii(s):
    # remove all non-ascii characters
    return s.encode('ascii', 'ignore').decode('ascii')
def preprocess_context(context):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_context = [word for word in context if word not in stop_words]
    lemmatized_context = [lemmatizer.lemmatize(word) for word in filtered_context]
    return lemmatized_context

def most_frequent_sense(instance, pos=None):
    synsets = wn.synsets(instance.lemma, pos)
    return synsets[0] if synsets else None

def lesk_algorithm(instance, context, pos=None):
    return lesk(context, instance.lemma, pos)

def bigram_features(instance, window_size = 2):
    start = max(0, instance.index - window_size)
    end = min(len(instance.context), instance.index + window_size + 1)
    bigrams = [' '.join(instance.context[i:i+2]) for i in range(start, end - 1)]     
    return ' '.join(bigrams) 

def create_sentece_pairs(instance):
    synsets = wn.synsets(instance.lemma)
    sentence_pairs = []
    for synset in synsets:
        gloss = synset.definition()
        context_sentence = ' '.join(str(word) for word in instance.context)  
        sentence_pair= f"{context_sentence} [SEP] {gloss}"
        sentence_pairs.append((synset.name(), sentence_pair))
    return sentence_pairs

def predict_sense(instance, model, tokenizer):
    sentence_pairs = create_sentece_pairs(instance)
    max_score = float('-inf')  
    predicted_synset = None  
    for synset, sentence_pair in sentence_pairs:  
        inputs = tokenizer(sentence_pair, return_tensors="pt", truncation=True, padding=True)  
        with torch.no_grad():  
            outputs = model(**inputs)  
        score = outputs.logits[0, 1].item()  
  
        if score > max_score:  
            max_score = score  
            predicted_synset = synset  
  
    return predicted_synset
def preprocess_context(context):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_context = []  
    for word in context: 
        word = word.lower() 
        if word not in stop_words:  
            # Split words using special symbols  
            split_words = re.split(r'\W+', word)  
  
            # Lemmatize the split words and add them to the processed context  
            for split_word in split_words:  
                lemma = lemmatizer.lemmatize(split_word)  
                processed_context.append(lemma) 
    return processed_context  

def find_lemma_position(row):  
    lemma = row["Lemma"]  
    processed_context = row["processed_context"]  
    try:  
        return processed_context.index(lemma)  
    except ValueError:  
        return -1   

def surrounding_words(row, window_size=2):   
    context = row["processed_context"]  
    lemma_index = row["lemma_pos"]  
  
    # Set the start and end indices based on the position of the lemma  
    if lemma_index == 0:  
        start = 0  
        end = min(len(context), 5)  
    elif lemma_index == 1:  
        start = 0  
        end = min(len(context), 5)  
    elif lemma_index == len(context) - 1:  
        start = max(0, lemma_index - 4)  
        end = lemma_index + 1  
    elif lemma_index == len(context) - 2:  
        start = max(0, lemma_index - 3)  
        end = min(len(context), lemma_index + 2)  
    else:  
        start = max(0, lemma_index - 2)  
        end = min(len(context), lemma_index + 3)  
  
    # Get the 5-word window around the lemma  
    five_word_window = context[start:end]  
  
    return ' '.join(five_word_window)
# Assuming 'df' is the DataFrame created in the previous step  

def bootstrapping(classifier, filtered_df, max_iter=10, threshold=0.9):  
    vectorizer = CountVectorizer(ngram_range=(1,2))   
    vectorizer.fit(filtered_df["bigram_context"].tolist())  # Fit the vectorizer on the entire dataset  
    seed_set_df = filtered_df.sample(n=5, random_state=42)  
    unlabeled_set_df = filtered_df.drop(seed_set_df.index)  
    seed_set = seed_set_df["bigram_context"].tolist()  
    seed_sense = seed_set_df["Numerical_Label"].tolist()  
    unlabeled_set = unlabeled_set_df["bigram_context"].tolist()  
    unlabeled_sense = unlabeled_set_df["Numerical_Label"].tolist()  
    for _ in range(max_iter):  
        X_labeled = vectorizer.transform(seed_set)  # Only transform the data, do not fit the vectorizer again  
        y_labeled = seed_sense  
        classifier.fit(X_labeled, y_labeled)  
        confident_set = []  
        confident_sense = []  
        for i, (instance, sense) in enumerate(zip(vectorizer.transform(unlabeled_set), unlabeled_sense)):  
            X_unlabeled = instance  
            prob_dist = classifier.predict_proba(X_unlabeled)[0]  
            most_probable_sense_index = prob_dist.argmax()  
            max_prob = prob_dist[most_probable_sense_index]  
            most_frequent_sense = classifier.classes_[most_probable_sense_index]  
            if max_prob >= threshold:  
                confident_set.append(unlabeled_set[i])  
                confident_sense.append(most_frequent_sense)  
        if not confident_set:  
            seed_set_df = filtered_df.sample(n=5, random_state=42)  
            unlabeled_set_df = filtered_df.drop(seed_set_df.index)  
            seed_set = seed_set_df["bigram_context"].tolist()  
            seed_sense = seed_set_df["Numerical_Label"].tolist()  
            unlabeled_set = unlabeled_set_df["bigram_context"].tolist()  
            unlabeled_sense = unlabeled_set_df["Numerical_Label"].tolist()  
            continue  
        seed_set.append(confident_set[0])  
        seed_sense.append(confident_sense[0])  
        unlabeled_set.remove(confident_set)  
        unlabeled_sense.remove(confident_sense[0])  
        print(f"Iteration {_+1}, seed_set size: {len(seed_set)}")  
      
    X = vectorizer.transform(filtered_df["bigram_context"].tolist())  # Only transform the data, do not fit the vectorizer again  
    y = filtered_df["Numerical_Label"]  
    y_pred = classifier.predict(X)  
    return classifier, metrics.accuracy_score(y, y_pred)  

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    correct_mfs_dev = 0
    correct_mfs_test = 0
    correct_lesk_dev = 0
    correct_lesk_test = 0  
    correct_BERT_dev = 0
    correct_BERT_test = 0
    for instance_id_dev, instance_dev in dev_instances.items():
        context = preprocess_context(instance_dev.context)
        mfs_result = most_frequent_sense(instance_dev)
        lesk_result = lesk_algorithm(instance_dev, context)
        if mfs_result and mfs_result.name() in convert_synset(dev_key[instance_id_dev]):
            correct_mfs_dev += 1
        if lesk_result and lesk_result.name() in convert_synset(dev_key[instance_id_dev]):
            correct_lesk_dev += 1
    accuracy_mfs = correct_mfs_dev / len(dev_instances)
    accuracy_lesk = correct_lesk_dev / len(dev_instances)
    print('Accuracy of Most Frequent Sense in Dev set: %f' % accuracy_mfs)
    print('Accuracy of Lesk Algorithm in Dev set: %f' % accuracy_lesk)
    for instance_id_test, instance_test in test_instances.items():
        context = preprocess_context(instance_test.context)
        mfs_result = most_frequent_sense(instance_test)
        lesk_result = lesk_algorithm(instance_test, context)
        if mfs_result and mfs_result.name() in convert_synset(test_key[instance_id_test]):
            correct_mfs_test += 1
        if lesk_result and lesk_result.name() in convert_synset(test_key[instance_id_test]):
            correct_lesk_test += 1
    accuracy_mfs = correct_mfs_test / len(test_instances)
    accuracy_lesk = correct_lesk_test / len(test_instances)
    print('Accuracy of Most Frequent Sense in Test set: %f' % accuracy_mfs)
    print('Accuracy of Lesk Algorithm in Test set: %f' % accuracy_lesk)
    #import BERT
    model_name = 'kanishka/GlossBERT'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    for instance_id_dev, instance_dev in dev_instances.items():
        predicted_synset = predict_sense(instance_dev, model, tokenizer)
        correct_sense_keys = dev_key[instance_id_dev]  
        correct_offsets = [wn.lemma_from_key(sense_key).synset().offset() for sense_key in correct_sense_keys]
        print(f"Predicted Synset: {predicted_synset}")
        print(f"Correct Synset: {convert_synset(dev_key[instance_id_dev])}")
        if predicted_synset == convert_synset(dev_key[instance_id_dev]):  
            correct_BERT_dev += 1
    accuracy_BERT = correct_BERT_dev / len(dev_instances)
    print('Accuracy of BERT Dev set: %f' % accuracy_BERT)
    for instance_id_test, instance_test in test_instances.items():
        predicted_synset = predict_sense(instance_test, model, tokenizer)
        correct_sense_keys = test_key[instance_id_test]  
        correct_offsets = [wn.lemma_from_key(sense_key).synset().offset() for sense_key in correct_sense_keys]
        print(f"Predicted Synset: {predicted_synset}")
        print(f"Correct Synset: {convert_synset(test_key[instance_id_test])}")
        if predicted_synset == convert_synset(test_key[instance_id_test]):  
            correct_BERT_test += 1
    accuracy_BERT = correct_BERT_test / len(test_instances)
    print('Accuracy of BERT Test set: %f' % accuracy_BERT)

    #bootstrapping
    lemmas = ["plan", "climate", "focus", "term", "path"]
    with open("bootstrapping.txt", "r") as file:
        content = file.read()

    #Split the content in into rows and remove the first row
    rows = content.split("\n")[1:]  

    data = []  
    for row in rows:  
        if row:  
            columns = row.split("|")[1:-1] 
            if len(columns) == 3: 
                lemma, context, label = columns  
                lemma = lemma.strip()  
                context = context.strip()  
                label = label.strip()  
                data.append((lemma, context, label))  
            else:  
                print(f"Skipping row with unexpected format: {row}") 
    
    # Define the columns  
    columns = ["Lemma", "Context", "Label"]  

    # Create a DataFrame  
    df_bootstrapping = pd.DataFrame(data, columns=columns)  
    df_numerical_labels = df_bootstrapping.copy()
    #Initialize the LabelEncoder
    encoder = LabelEncoder()
    df_numerical_labels['Label'] = encoder.fit_transform(df_numerical_labels['Label'])
    df_numerical_labels.rename(columns={"Label": "Numerical_Label"}, inplace=True)  
    # Create a new DataFrame 'y' containing only the numerical labels  
    y = pd.DataFrame(df_numerical_labels["Numerical_Label"]) 
    df_numerical_labels['processed_context'] = df_numerical_labels['Context'].apply(lambda x: preprocess_context(x.split()))  
    # Apply the function to each row and create a new column 'Lemma_Index'  
    df_numerical_labels["lemma_pos"] = df_numerical_labels.apply(find_lemma_position, axis=1)   
    df_numerical_labels["bigram_context"] = df_numerical_labels.apply(surrounding_words, axis=1)  
    df_numerical_labels["bigram_context"] = df_numerical_labels.apply(surrounding_words, axis=1) 

    for lemma in lemmas:
        filtered_df = df_numerical_labels[df_numerical_labels["Lemma"] == lemma]
        classifier = MultinomialNB()
        naive_bayes, accuracy_score = bootstrapping(classifier, filtered_df, max_iter=10)
        print("Accuracy for lemma", lemma, "is:", accuracy_score)