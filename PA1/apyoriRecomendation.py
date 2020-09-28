from apyori import apriori
import csv
import pandas as pd
import pickle


def maxLength():
    '''
    Find and returns the length of the longest row
    '''
    index = 0
    temp = 0
    with open("browsing-data.txt", "r") as csvfile:
        hw1reader = csv.reader(csvfile, delimiter=' ')
        for row in hw1reader:
            new_str = ''
            print("Length", len(row))

            if index == 0:
                temp = len(row)
            elif len(row) > temp:
                temp = len(row)
            index += 1
            print("Index:", index)
            for x in range(len(row)):
                if (row[x] != ''):
                    new_str += row[x]
                    new_str += ','
            print(new_str)

            print("#" * 30)
            print("#" * 30)

    print("Maximum Length:", temp)
    return temp


def addDelimitors(maxLength):
    '''
    Adds missing comas to rows that have less columns than maxLength
    '''

    index = 0
    new_file = ''

    with open('browsing-data.txt') as csvfile:
        h1reader = csv.reader(csvfile, delimiter=' ')
        for row in h1reader:
            new_str = ''
            print("Length:", row)
            index += 1
            print("Index:", index)
            for x in range(len(row)):
                if (row[x] != ''):
                    new_str += row[x]
                    new_str += ','

            print(new_str)
            if len(row) < maxLength:
                # diff = maxLength - len(row)
                for x in range(len(row), maxLength):
                    new_str += ','
            print(new_str)
            new_file += new_str + "\n"
            print("#" * 30)
            print("#" * 30)

    print(maxLength)
    print(new_file)

    with open("preProcessedData.txt", "w+") as f:
        f.write(new_file)

    # file = open("preProcessedData.txt", "r")
    # print(file.read())

    # with open("preProcessedData.txt", "r") as csvfile:
    #     readcsv = csv.reader(csvfile, delimiter=",")
    #     for row in readcsv:
    #         print(row)
    #         print(len(row))


def process(length):
    '''
    Creates array with all the preProcessedData
    Passes results array to the apriori algorithm
    Sorts results by length then name

    Throughout the entire processs pickle files will be created as 'savespoints'
        in case of computer issues during the long runtime
    '''
    data = pd.read_csv("preProcessedData.txt", header=None)
    data.head()

    data.dropna()
    data.head()
    data.info()
    records = []
    rows = data.shape[0]
    cols = data.shape[1]
    print(rows)
    print(cols)

    for i in range(0, rows):
        print(i, end="\r")
        temp = []
        for j in range(0, length):
            temp.append(str(data.values[i, j]))
        if i % 10000 == 0:
            with open("records{}.pickle".format(i), 'wb') as handle:
                pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        records.append(temp)

    with open("records.pickle", 'wb') as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Records len: ", len(records))
    print("Support: ", 100/rows)
    # records = []
    # with open("records.pickle", "rb") as handle:
    #     records = pickle.load(handle)

    hw1rules = apriori(records, min_support=(100/rows),
                       min_confidence=0.5, min_lift=4, min_length=2, max_length=3)
    print("done")

    hw1results = list(hw1rules)
    with open("rules.pickle", 'wb') as handle:
        pickle.dump(hw1results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(hw1results))

    # hw1results = []
    # with open("rules.pickle", "rb") as handle:
    #     hw1results = pickle.load(handle)

    # for item in hw1results:
    #     pair = item[0]
    #     items = [x for x in pair]
    #     print("Rule:" + items[0] + " -> " + items[1])
    #     print("Support: " + str(items[1]))
    #     print("Confidence: " + str(item[2][0][2]))
    #     print("Lift: " + str(item[2][0][3]))
    #     print("=" * 30)

    # print("#"*40)
    # print("#"*40)

    # hw1results.sort(key=lambda x: x[2][0][2], reverse=True)
    #####################################################################
    ## Change the == 2 to the length of the support you want to see from the results
    #####################################################################
    lenfilter = filter(lambda x: len([y for y in x[0]][0:]) == 2, hw1results)
    filteredResults = list(lenfilter)
    filteredResults.sort(key=lambda x: (x[2][0][2], [y for y in x[0]]), reverse=True)
    # filteredResults.sort(key=lambda x: len([y for y in x[0]][1:]), reverse=True)

    for i in range(15):
        item = filteredResults[i]
        # print(item)
        pair = item[0]
        items = [x for x in pair]
        print("Rule:" + items[0] + " -> " + str(items[1:]))
        print("Support: " + str(items[1:]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=" * 30)


if __name__ == "__main__":
    length = maxLength()
    addDelimitors(length)
    process(length)
