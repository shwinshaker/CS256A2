#!/bin/python

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")

    """
    - Spelling correction
        Largest gain examples:
            awsome -> awesome

    - lengthening reduction (take effect only if using spelling correction)
        gain: about 0.002

    """
    from symspellpy.symspellpy import SymSpell, Verbosity
    import os
    import re
    class SpellCorrector():
        def __init__(self, max_edit_distance_dictionary=2, prefix_length=7):
            self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
            # load dictionary
            dictionary_path = os.path.join(os.path.dirname('.'),
                                           "frequency_dictionary_en_82_765.txt")
            term_index = 0  # column of the term in the dictionary text file
            count_index = 1  # column of the term frequency in the dictionary text file
            if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index):
                raise("Dictionary file not found")

            # manually
            # this works. about 0.003 up
            # self.corr_dict = {"awsome": "awesome"}

        def reduce_lengthening(self, text):
            # not work
            pattern = re.compile(r"(.)\1{2,}")
            return pattern.sub(r"\1\1", text)

        def strip_punc(self, word):
            # not work
            return re.sub(r"[\-\_\.\!]$", "", word)

        def __call__(self, word):
            word = self.reduce_lengthening(word)
            # if word in self.corr_dict:
            #     word = self.corr_dict[word]
            if len(word) > 2 and "'" not in word:
                suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, 2)
                if suggestions:
                    return suggestions[0].term
            return word
            # word = self.strip_punc(word)

            # if word in self.corr_dict:
            #     return self.corr_dict[word]
            # return word

    """
    - tokenization:
        deal with punctuations better
            eg. aren't -> are n't
                it's -> it 's
        The default tokenizor will treat "aren't" as "aren" and neglect "'t".
        This is detrimental because negative statement becomes neutral.

    - lemmatization
        group words by its original form
            eg. rocks -> rock
        gain only when lemmatizing nouns
        lemmatizing verbs is detrimental

    """
    from nltk import word_tokenize
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk import pos_tag
    class LemmaTokenizer(object):
        def __init__(self):
            self.lemma = WordNetLemmatizer()
            # self.tokenizer = RegexpTokenizer(r"(?u)\b\w+[\'\!\*]*\w*\b")
            # self.wnl = PorterStemmer()
            self.corrector = SpellCorrector()

        def lemmatize(self, token, tag):  
            if tag[0].lower() in ['n','v']:  
                return self.lemma.lemmatize(token, tag[0].lower())  
            return token

        def __call__(self, sentence):
            return [self.lemma.lemmatize(self.corrector(t.lower()), 'n') for t in word_tokenize(sentence)]
            # return [self.lemma.lemmatize(t, 'v') for t in s]
            # return [self.lemmatize(self.corrector(t.lower()), tag) for t, tag in pos_tag(word_tokenize(sentence))]    
            # return [self.wnl.stem(t) for t in word_tokenize(articles)] 
            # return [self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(articles)]

    """
    - tfidf:
        modify the word frequency based on its occurence through the whole corpus.
        frequencies of frequently used words like 'the' 'a' will be diminished

    - ngram:
        introduce more features, and capture relations between words

    """  
    # from sklearn.feature_extraction.text import CountVectorizer
    # sentiment.count_vect = CountVectorizer(ngram_range=(1,2))
    from sklearn.feature_extraction.text import TfidfVectorizer
    # todo - better if ignore numbers?
    # caution - token_pattern will be overriden by tokenizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,3),
                                           norm=None,
                                           # token_pattern=r"(?u)\b[a-zA-Z]+[\'\!\*]*\b",
                                           # token_pattern=r"(?u)\b\w+[\'\!\*]*\w*\b",
                                           sublinear_tf=True,
                                           stop_words=[],
                                           tokenizer=LemmaTokenizer())
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    # print(type(sentiment.trainX))
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def preprocessing():
    pass

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')


    # print("\nReading unlabeled data")
    # unlabeled = read_unlabeled(tarfname, sentiment)
    # print("Writing predictions to a file")
    # write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)

    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
