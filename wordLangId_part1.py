#####################################################################
# Jonathan Wachholz
# jhw190002
# HW 4
#
# A simple program that reads in 3 different language files located
# in the data/ folder, tokenizes them, and creates 3 sets of Unigram
# and Bigram dictionaries, mapping the N-Grams to their occurance count.
# These Dictionaries are then pickled for use in part 2.
#
# Part 2 will open the pickled dictionaries and use them to predict the
# language for the lines in a given input test file. The predictions will
# then be written to an output file, and the output file will be compared
# against the true results for the given input test file for accuracy.
#
#####################################################################

import operator
import os
import pickle
from time import perf_counter

import nltk
from nltk.util import ngrams


def openFile(filepath: str, mode: str, encoding: str = None):
    """Function to open a file path that works cross platform"""
    return open(os.path.join(os.getcwd(), filepath), mode, encoding=encoding)


def getRawText(filepath: str, mode: str, encoding: str = None, skipLines: int = 0) -> str:
    """"
    Function to open a text file and return the raw text from that file...
    :param filepath: The path location of the file
    :param skipLines: The number of lines of the text file to skip over when reading text from the file

    :returns The raw text of the file
    """
    text = str()
    with openFile(filepath, mode, encoding) as inFile:
        for i in range(skipLines):
            inFile.readline()

        text += inFile.read()
    return text


def prog1Function(filepath: str, DEBUG = True):
    """
    A Function to take in a given input filepath and extract the raw text from that file.
    The text will be tokenized, then converted to Uni and Bi Gram list, which will then be converted to
    Uni and Bi Gram dictionaries mapping the N-grams to their occurance count...
    :returns UniGramDict, BiGramDict
    """
    # a and b
    text = getRawText(filepath, "r", encoding="utf-8").replace("\n", " ")

    # c
    tokens = nltk.tokenize.word_tokenize(text)
    #tokens = nltk.regexp_tokenize(text, r"\w+")

    if DEBUG:
        print(f"The length of TokensList is: {len(tokens)}")
    # d
    biGramsList = list(nltk.ngrams(tokens, 2))
    if DEBUG:
        print(f"The length of BiGramsList is: {len(biGramsList)}")
    # e
    uniGramsList = list(nltk.ngrams(tokens, 1))
    if DEBUG:
        print(f"The length of UniGramsList is: {len(uniGramsList)}")
    # f
    # biGramsDict = {tup: biGramsList.count(tup) for tup in biGramsList}
    biGramsDict = dict()
    for tup in biGramsList:
        if tup not in biGramsDict.keys():
            biGramsDict[tup] = 1
        else:
            biGramsDict[tup] += 1

    # g
    # uniGramsDict = {word: uniGramsList.count(word) for word in uniGramsList}
    uniGramsDict = dict()
    for elem in uniGramsList:
        if elem not in uniGramsDict.keys():
            uniGramsDict[elem] = 1
        else:
            uniGramsDict[elem] += 1

    return uniGramsDict, biGramsDict


def writePickle(filepath: str, object, objName: str = ""):
    """Function to open up and write an object to a pickle file..."""
    with openFile(filepath, mode="wb") as pickleFile:
        pickle.dump(object, pickleFile)
        print(f"Successfully wrote out the obj: {objName} to \'{filepath}\'")


if __name__ == '__main__':
    # i
    t1 = perf_counter()
    engUnigramsDict, engBigramsDict = prog1Function(filepath="data/LangId.train.English")
    frenUnigramsDict, frenBigramsDict = prog1Function(filepath="data/LangId.train.French")
    italUnigramsDict, italBigramsDict = prog1Function(filepath="data/LangId.train.Italian")

    t2 = perf_counter()
    print("Time to generate all dicts: ", t2 - t1,"seconds")

    writePickle("data/engUniGramDict.pickle", engUnigramsDict, "engUnigramsDict")
    writePickle("data/frenUniGramDict.pickle", frenUnigramsDict, "frenUnigramsDict")
    writePickle("data/italUniGramDict.pickle", italUnigramsDict, "italUnigramsDict")

    writePickle("data/engBigramsDict.pickle", engBigramsDict, "engBigramsDict")
    writePickle("data/frenBigramsDict.pickle", frenBigramsDict, "frenBigramsDict")
    writePickle("data/italBigramsDict.pickle", italBigramsDict, "italBigramsDict")

    t3 = perf_counter()
    print("Time for pickling:", t3 - t2,"seconds")

    dictList = [engUnigramsDict, engBigramsDict, frenUnigramsDict, frenBigramsDict , italUnigramsDict, italBigramsDict]
    for elem in dictList:
        sortedDictList = sorted(elem.items(), key=operator.itemgetter(1), reverse=True)
        print(f"\t{sortedDictList[:25]}")

    print()
