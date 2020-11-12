from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")

stop = stopwords.words("english")


def readFortuneFiles():
    r'''
    Read training data and test data from files
    Args:
        None
    Return:
        None, updates values in global variables
    '''
    global x, y, z, predData, predLabels

    x = pd.read_csv(r"fortune-cookie-data/traindata.txt", header=None)
    y = pd.read_csv(r"fortune-cookie-data/trainlabels.txt", header=None)
    z = pd.read_csv(r"fortune-cookie-data/stoplist.txt", header=None)
    predData = pd.read_csv(r"fortune-cookie-data/testdata.txt", header=None)
    predLabels = pd.read_csv(
        r"fortune-cookie-data/testlabels.txt", header=None)


def processFortune():
    r'''
    Preprocess training data and test data
    Args:
        None, reads from global variables
    Return:
        None, updates values in global variables
    '''
    global x, y, z, predData, predLabels

    # Training data size
    rangeX = x.size
    # Test data size
    rangePred = predData.size

    # Rename columnms
    x.columns = ["Data"]
    y.columns = ["Label"]
    z.columns = ["Stop"]
    predData.columns = ["Data"]
    predLabels.columns = ["Label"]

    # Combine Training and test data
    dX = pd.concat([x, predData])

    # List of stop words
    stopwords = z["Stop"].tolist()

    # Create Tokenized Data
    dX["Filt_Data"] = dX["Data"].apply(lambda i: " ".join(
        [word for word in i.split() if word not in (stopwords)]))
    del dX["Data"]
    dX["Tokenized_Data"] = dX.apply(
        lambda row: nltk.word_tokenize(row["Filt_Data"]), axis=1)

    # Training labels to list
    y = y["Label"].tolist()
    # Test labels to list
    predLabels = predLabels["Label"].tolist()

    # Training Data - TFIDF, find most frequent words in sentances
    v = TfidfVectorizer()

    Tfidf = v.fit_transform(dX["Filt_Data"])

    df1 = pd.DataFrame(Tfidf.toarray(), columns=v.get_feature_names())

    # Seperate Training and Test Data
    x = df1[0:rangeX]
    predData = df1[rangeX:rangeX+rangePred]


def fitFortune():
    r'''
    Train a perceptron with training data and then predict
        classificaiton of test_data
    Args:
        None
    Return:
        None, print results to output.txt
    '''

    global x, y, z, predData, predLabels

    # Declare Perceptron settings for binary classification
    ppn = Perceptron(max_iter=20, eta0=1, random_state=0, verbose=1)

    with open("output.txt", "w") as f:
        for i in range(20):
            ppn.partial_fit(x, y, classes=np.unique(y))
            y_pred = ppn.predict(predData)
            score = accuracy_score(predLabels, y_pred)
            mistakes = int(len(y_pred) - score * len(y_pred))
            print("iteration-{} {}".format(i+1, mistakes), file=f)


def split(word: str):
    r'''
    Create a list from a string

    Args:
        word: string to turn to array
    Return:
        list with string in it
    '''
    return list(word)


def pre_process(file: str) -> tuple:
    r'''
    Pre process multi class file into data and labels

    Args:
        file: path to file location
    Return:
        tuple(classData, classLabels)
    '''
    with open(file, "r") as f:
        lines = f.readlines()

    i = 0
    for line in lines:
        l = re.split(r"\t+", line)

        if len(l) > 2:

            l1 = split(l[1][2:len(l[1])])
            dsl = len(l1)
            if i == 0:
                k1 = np.array(l1)
                k = k1
                label1 = np.array(l[2])
                label = label1
            else:
                k1 = np.array(l1)
                k = np.append(k, k1)
                label1 = np.array(l[2])
                label = np.append(label, label1)
            i = i+1
        print(i, end="\r")
    k2 = k.reshape(i, dsl)
    dataframe = pd.DataFrame.from_records(k2)
    return (dataframe, label)


def implement_perceptron(train_data: pd.DataFrame, train_label: np.array,
                         test_data: pd.DataFrame, test_label: np.array):
    r'''
    Perform multi class classification on test_data after fitting the
        perceptron to the training_data
    Args:
        train_data: pandas dataframe with training data
        train_label: numpy array with training labels
        test_data: pandas dataframe with test data
        test_label: numpy array with test labels
    Return:
        None, prints results to output.txt
    '''

    ppn = Perceptron(max_iter=50, eta0=0.1, random_state=0, verbose=1)
    with open("output.txt", "a") as f:
        for i in range(20):
            ppn.partial_fit(train_data, train_label,
                            classes=np.unique(train_label))
            test_pred = ppn.predict(test_data)
            train_pred = ppn.predict(train_data)

            testScore = accuracy_score(test_label, test_pred)
            trainScore = accuracy_score(train_label, train_pred)

            print("iteration-{} {} {}".format(i+1, trainScore, testScore))


if __name__ == "__main__":
    # Binary classification
    readFortuneFiles()
    processFortune()
    fitFortune()

    # MultiClass classification
    train = "OCR-data/OCR-data/ocr_test.txt"
    test = "OCR-data/OCR-data/ocr_train.txt"
    train_data, train_labels = pre_process(train)
    test_data, test_labels = pre_process(test)

    implement_perceptron(train_data, train_labels, test_data, test_labels)
