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

import os
import pickle
from typing import List, Tuple, Dict
import nltk


def openFile(filepath: str, mode: str, encoding: str = None):
    """Function to open a file path that works cross platform"""
    return open(os.path.join(os.getcwd(), filepath), mode, encoding=encoding)


def openPickle(filepath: str, objName: str = ""):
    """Function to open up and return a pickled object from a pickle file..."""
    with openFile(filepath, mode="rb") as pickleFile:
        obj = pickle.load(pickleFile)
        print(f"Successfully read in the obj: {objName} from \'{filepath}\'")
        return obj


def writePickle(filepath: str, object, objName: str = ""):
    """Function to open up and write an object to a pickle file..."""
    with openFile(filepath, mode="wb") as pickleFile:
        pickle.dump(object, pickleFile)
        print(f"Successfully wrote out the obj: {objName} to \'{filepath}\'")


def isEOF(file):
    """Function to check if the given file / stream object has reached its End Of File position..."""
    currentPos = file.tell()  # save current position
    file.seek(0, os.SEEK_END)
    endPos = file.tell()  # find the size of file
    file.seek(currentPos, os.SEEK_SET)
    return currentPos == endPos


def printList(lst: list, endStr: str = "\n"):
    """Function to print out the items in a list individually,
    with the option to modify the endString attribute of print()..."""
    for item in lst:
        print(item, end=endStr)
    print()


def compareProbabilities(trueFilePath: str, predFilePath: str):
    """Function to open up 2 files to compare their results... It will print the results at the end,
        displaying the amount of correct results, total results, the accuracy, and printing the incorrect items found..."""
    lineNum = 0
    correctLines = 0
    incorrectLines = 0
    incorrectLinesList = []
    incorrectLineNums = []
    with openFile(trueFilePath, mode="r") as trueFile, openFile(predFilePath, mode="r") as predFile:
        trueEOF_flag = False
        predEOF_flag = False
        while not trueEOF_flag and not predEOF_flag:
            if lineNum == 300:
                print()

            while not trueEOF_flag:
                trueEOF_flag = isEOF(trueFile)
                trueLine = trueFile.readline().strip().lower()
                if not trueLine.__eq__(""):
                    break

            while not predEOF_flag:
                predEOF_flag = isEOF(predFile)
                predLine = predFile.readline().strip().lower()
                if not predLine.__eq__(""):
                    break

            if trueEOF_flag or predEOF_flag:
                break

            lineNum += 1
            if not trueLine == predLine:
                incorrectLines += 1
                incorrectLinesList.append(f"\tLine {lineNum}: Real = \'{trueLine}\', Predicted = \'{predLine}\'")
                incorrectLineNums.append(lineNum)
            else:
                correctLines += 1

    print(f"Correct: {correctLines} \nTotal: {lineNum} \nAccuracy: {correctLines / lineNum} \nIncorrect Items:")
    printList(incorrectLineNums, endStr=" ")
    printList(incorrectLinesList)


# (b + 1) / (u + v) where b is
# the bigram count, u is the unigram count of the first word in the bigram, and v is the total vocabulary
# size (add the lengths of the 3 unigram dictionaries).
#
def calcProbability(testFilePath: str, outFilePath: str, dictTuples: List[Tuple[Dict, Dict, str]], DEFAULT: str = "",
                    DEBUG: bool = False):
    """
    Function to take in a test file and estimate the language probability based on a given list of language dictionaries...
    :param testFilePath = The test file in which to read in and predict the language of, line by line.
    :param outFilePath = The file path to write the prediction results out to.
    :param dictTuples = a List of Tuples in the form of: (UnigramDict, BiGramDict, Language Name)
            ex. dictsTuples = [(engUnigramsDict, engBigramsDict, "English"), ...]
    :param DEFAULT = In the case of multiple languages having equally highest probabilities, will default to this language...
    :param DEBUG = Set to true to enable debug printing...
    """
    if len(dictTuples) < 1:
        print("There was an error... An empty list was passed as an argument... \n\t"
              "This function needs a list of Tuples in the form: \n\t\t"
              "List[(UnigramDict, BigramDict, \"Dictionary Language Name\"), ...]")
        exit(-1)

    lineNum = 0
    with openFile(testFilePath, mode="r", encoding="utf-8") as inFile, openFile(outFilePath, mode="w", encoding="utf-8") as outFile:
        vocabSize = 0
        for tup in dictTuples:
            vocabSize += len(tup[0].keys())
        for line in inFile:
            lineNum += 1
            if lineNum == 277:
                print()
            testTokens = nltk.word_tokenize(line)
            testBiGrams = list(nltk.ngrams(testTokens, 2))
            #testBiGrams.insert(0, ("START", testTokens[0]))
            if len(testTokens) == 1:
                testBiGrams = [(".", testTokens[0])]
            # Probabilities using LaPlace smoothing
            probLaPlace = 1
            probsLaPlaceList = [1 for tup in
                                dictTuples]  # Should instantiate a list of values 1 for the length of the dicts parameter
            nList = [1 for tup in dictTuples]
            dList = [1 for tup in dictTuples]
            for testBigram in testBiGrams:
                # dicts[tuple( UniGramDict, BiGramDict, str)
                for i in range(len(dictTuples)):
                    # UniGramsDict = dicts[i][0]
                    # BiGramsDict  = dicts[i][1]

                    # n = dictTuples[i][1].get(testBigram) if testBigram in dictTuples[i][1] else 0
                    nList[i] = dictTuples[i][1].get(testBigram) if testBigram in dictTuples[i][1] else 0

                    # d = dictTuples[i][0].get(testBigram[0]) if testBigram[0] in dictTuples[i][0] else 0
                    dList[i] = dictTuples[i][0].get(testBigram[0]) if testBigram[0] in dictTuples[i][0] else 0

                    # probLaPlace *= ((n + 1) / (d + vocabSize))
                    probsLaPlaceList[i] *= ((nList[i] + 1) / (dList[i] + vocabSize))

            # Printing the results for the line
            maxProb = -1
            maxProbIndex = -1
            for i in range(len(dictTuples)):
                # print(f"\tProbability {dicts[i][2]} = {probsLaPlaceList[i]}")
                if probsLaPlaceList[i] > maxProb:
                    maxProb = probsLaPlaceList[i]
                    maxProbIndex = i
            maxProbsLang = dictTuples[maxProbIndex][2]
            multiMaxProbLangFlag = False
            for i in range(len(dictTuples)):
                if i == maxProbIndex:
                    continue
                if probsLaPlaceList[i] == maxProb:
                    maxProbsLang += f" / {dictTuples[i][2]}"
                    multiMaxProbLangFlag = True

            if DEBUG:
                print(f"Line {lineNum}: {line.strip()}"
                      f"\n\t\t{maxProbsLang}")
            if maxProb == 0:
                maxProbsLang += " (0 Probability)"
            elif multiMaxProbLangFlag:
                maxProbsLang += f" ({maxProb} Probability)"
            if not DEFAULT.__eq__("") and multiMaxProbLangFlag:
                maxProbsLang = DEFAULT

            outFile.write(f"{lineNum} {maxProbsLang}\n")


if __name__ == '__main__':
    # a
    engUnigramsDict = dict(openPickle("data/engUniGramDict.pickle"))
    engBigramsDict = dict(openPickle("data/engBigramsDict.pickle"))
    frenUnigramsDict = dict(openPickle("data/frenUniGramDict.pickle"))
    frenBigramsDict = dict(openPickle("data/frenBigramsDict.pickle"))
    italUnigramsDict = dict(openPickle("data/italUniGramDict.pickle"))
    italBigramsDict = dict(openPickle("data/italBigramsDict.pickle"))

    dictsList = [(engUnigramsDict, engBigramsDict, "English"),
                 (frenUnigramsDict, frenBigramsDict, "French"),
                 (italUnigramsDict, italBigramsDict, "Italian")]

    testFile = "data/LangId.test"
    predFile = "data/LangId.predictions"
    soluFile = "data/LangId.sol"
    calcProbability(testFile, predFile, dictsList, DEFAULT="")

    compareProbabilities(soluFile, predFile)
