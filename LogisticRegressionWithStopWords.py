# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:23:46 2017

@author: Mohanakrishna
"""

import numpy as np
import os
import re
import math
import sys

ETA = 0.001
Lamda = 3

global hamcorrect
global hamwrong
global spamcorrect
global spamwrong

wordsList = {}
spamWordsList=[]
hamWordList=[]
fileTokenDict={}
weightsDict ={}
testFileTokenDict={}
testWeightsDict ={}
fileData = []
tokenCountDict = {}
weightVector={}

def Sigmoid(x):
    d = 1+np.exp(-x)
    return (1/d)

def getAllWords(path,stopWordsPath=None):
    hamPath = path +'/ham'
    spamPath = path +'/spam'
    hamDirectory = os.listdir(hamPath)
    spamDirectory = os.listdir(spamPath)
    spamSection =""
    hamSection =""
#    spamData = []
    stopWords = []
    if stopWordsPath is not None:
        f = open(stopWordsPath,'r',encoding='iso-8859-1')
        regex = re.compile('\\W*')
        stopWords = regex.split(f.read())
        f.close()
    for file in hamDirectory:
        hamFilePath = hamPath +'/' + file
        f=open(hamFilePath,'r',encoding='iso-8859-1')
        readSection = f.read()
        hamSection += readSection
        regex = re.compile('\\W*')
        #hamtokens = re.split(' ', readSection)
        hamTokens = re.split(r'\W*', readSection)
        #hamtokens = regex.split(readSection)
        hamTokensList = [singleToken.lower() for singleToken in hamTokens if (len(singleToken) >1 and singleToken not in stopWords)]
        store = {}
        error = 0
        for token in hamTokensList:
            if token in store:
                store[token] += 1.0
            else:
                store[token] = 1.0

            if token not in weightVector:
                weightVector[token] = 0

        fileData.append({'fileName':file,'text':readSection,'fileTokens':store,'class':1})
        f.close()

    for file in spamDirectory:
        spamfilePath = spamPath + '/' + file
        f = open(spamfilePath, 'r',encoding='iso-8859-1')
        readSection = f.read()
        spamSection+=readSection

        #spamtokens = re.split(' ', readSection)
        spamTokens = re.split(r'\W*', readSection)
        #spamtokens= regex.split(readSection)
        spamTokensList = [singleToken.lower() for singleToken in spamTokens if (len(singleToken) >1 and singleToken not in stopWords)]
        store = {}
        for token in spamTokensList:
            if token in store:
                store[token] += 1.0
            else:
                store[token] = 1.0

            if token not in weightVector:
                weightVector[token] = 0

        fileData.append({'fileName':file,'text':readSection,'fileTokens':store,'class':0})
        f.close()
        
def GradientDescent():
#    keys = weightVector.keys()
    for i in range(1,100):
        err= updateError()
        updateWeights()

def updateError():
    error=0
    for every_file in fileData:
        keys = every_file["fileTokens"]
        sum = 1
        for token in keys:
            sum += keys[token]*weightVector[token]
        every_file["error"] = Sigmoid(sum)
        error+= every_file["error"]
    print(error)
    return error

def updateWeights():
    keys = weightVector.keys()
    for token in keys:
        sum = 0
        for every_file in fileData:
            tokens = every_file["fileTokens"]
            trueValue = every_file["class"]
            if token in tokens:
               sum +=tokens[token]*(trueValue-every_file["error"])

        weightVector[token]+= ((sum*ETA)-(ETA*Lamda *weightVector[token]))
        
def Test(path):
    hamPath = path + '/ham'
    spamPath = path + '/spam'
    hamDirectory = os.listdir(hamPath)
    spamDirectory = os.listdir(spamPath)
    spamSection = ""
    hamSection = ""
    keys = weightVector.keys()
    hamcorrect=0
    spamcorrect=0
    hamwrong=0
    spamwrong=0
    for every_file in hamDirectory:
        hamFilePath = hamPath + '/' + every_file
        f = open(hamFilePath, 'r',encoding='iso-8859-1')
        readSection = f.read()
        regex = re.compile('\\W*')
        hamTokens = re.split(r'\W*', readSection)
        hamTokensList = [singleToken.lower() for singleToken in hamTokens if len(singleToken) > 1]
        sum = 0
        store = {}
        for token in hamTokensList:
            if token in store:
                store[token] += 1.0
            else:
                store[token] = 1.0

        f.close()
        for token in hamTokensList:
            if token in weightVector:
                sum+=weightVector[token]*store[token]
        sum =Sigmoid(sum)
        if sum > 0.5:
            hamcorrect += 1
        else:
            hamwrong += 1
#    totalhamcount = hamcorrect+hamwrong
#    print("Ham Correct: ",hamcorrect)
#    print("Wrong: ",hamwrong)

    for file in spamDirectory:
        spamfilePath = spamPath + '/' + file
        f = open(spamfilePath, 'r',encoding='iso-8859-1')
        readSection = f.read()
        regex = re.compile('\\W*')
        spamTokens = re.split(r'\W*', readSection)
        spamTokensList = [singleToken.lower() for singleToken in spamTokens if len(singleToken) > 1]
        sum = 0
        store = {}
        for token in spamTokensList:
            if token in store:
                store[token] += 1.0
            else:
                store[token] = 1.0
        f.close()
        
        for token in spamTokensList:
            if token in weightVector:
                sum+=weightVector[token]*store[token]
        sum =Sigmoid(sum)
        if sum <= 0.5:
            spamcorrect += 1
        else:
            spamwrong += 1
#
#    print("Spam Correct: ",spamcorrect)
#    print("Wrong",spamwrong)
    return hamcorrect,hamwrong,spamcorrect,spamwrong


#trainData = sys.argv[1]
#testData = sys.argv[2]

#trainData = "hw2_train/train"
#testData = "hw2_test/test"
#stopWordsPath= "stopwords.txt"

def mainfn():

    if len(sys.argv) != 3:
#        print len(sys.argv)
        print("Incorrect Arguments !!!. Please specify input as : python LogisticRegression.py <train-data-path> <test-data-path>")
        sys.exit(1)

    trainData = sys.argv[1]
    testData = sys.argv[2]
#    stopWordsPath= "stopwords.txt"
    
#    stopWords = stopWordsPath
#    getAllWords(trainData,stopWordsPath)
#    GradientDescent()
#    sum=10
#    result = Test(testData) 
#    print("Logistic Regression without stop words:")
#    print("Number of ham files classified successsfully : ",result[0])  # correct ham
#    print("Ham files not classified successsfully: ",result[1]) # wrong ham 
#    print("Number of spam files classified successsfully: ", result[2]) # correct spam
#    print("Spam files not classified successsfully: ", result[3]) # wrong spam
#    sum += result[0] + result[2]
#    tsum = result[0] + result[2]  +result[1] + result[3]
#    r=(100*sum/float(tsum))
#    print("Accuracy: ",r)
#    print("\n")
    

#    stopWords = None

    getAllWords(trainData)
    GradientDescent() # calculates the gradient descent
    sum=10
    result = Test(testData) # calculate the result
    print("Logistic Regression with stop words:")
    print("Number of ham files classified successsfully : ",result[0])  # correct ham
    print("Number of ham files not classified successsfully: ",result[1]) # wrong ham 
    print("Number of spam files classified successsfully: ", result[2]) # correct spam
    print("Number of spam files not classified successsfully: ", result[3]) # wrong spam
    sum += result[0] + result[2]
    tsum = result[0] + result[2]  +result[1] + result[3]
    r=(100*sum/float(tsum))
    print("Accuracy: ",r)
#
if __name__ == "__main__":
	mainfn()
    



