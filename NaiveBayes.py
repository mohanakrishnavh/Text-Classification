# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:40:17 2017

@author: Mohanakrishna
"""

from __future__ import division
import os
import re
from math import log
import sys

#Read Folders
def folderText(classifier,path):
  section = " "
  directory = os.listdir(path)
  count = 0
  for everyfile in directory:
    count += 1
    filesPath = path + '/' + everyfile
    f = open(filesPath, 'r',encoding='iso-8859-1')
    section += f.read()
    #print(spamCount)
    #print(spamSection)
  return count,section

def fnPrior(path,classifier):
  spamham = folderText(classifier,path)
  count = spamham[0]
  text = spamham[1]
#  print(SpamCount,SpamText)
  return count,text


#Calculating Conditional Probalities
def spamConditionalProbablity(hamSection,stopWordsPath):
    global totalHamTokenCount
    totalHamTokenCount = 0
    noshowHamProb =1
    hamWordsList = Test(hamSection,stopWordsPath)
    hamDictionary ={}
    for word in hamWordsList:
        if word in hamDictionary:
            hamDictionary[word] += 1.0
        else:
            hamDictionary[word] = 1.0
    for key,value in hamDictionary.items():
            #print key, hamDictionary[k]
        totalHamTokenCount += hamDictionary[key] + 1.0
    hamProbList = []
    hamProbDictionary ={}
    for key,value in hamDictionary.items():
        tokenHamProb = (hamDictionary[key] + 1.0) / totalHamTokenCount
        hamProbDictionary[key] = tokenHamProb
        hamProbList.append(tokenHamProb)
    noshowHamProb = noshowHamProb/totalHamTokenCount
    return hamDictionary,hamProbList,hamProbDictionary,noshowHamProb

#Calculating Conditional Probality of Spam
def hamConditionalProbablity(spamSection,stopWordsPath):
    global totalSpamTokenCount
    totalSpamTokenCount = 0
    noshowSpamProb =1
    spamWordsList = Test(spamSection,stopWordsPath)
    spamDictionary ={}
    for word in spamWordsList:
        if word in spamDictionary:
            spamDictionary[word] += 1.0
        else:
            spamDictionary[word] = 1.0

    for key,value in spamDictionary.items():
        totalSpamTokenCount += spamDictionary[key] +1.0
    hamProbDictionary ={}
    spamProbList = []
    for key,value in spamDictionary.items():
        tokenSpamProb = (spamDictionary[key] + 1.0) / totalSpamTokenCount
        hamProbDictionary[key] = tokenSpamProb
        spamProbList.append(tokenSpamProb)
    noshowSpamProb = noshowSpamProb/totalSpamTokenCount
    return spamDictionary,spamProbList,hamProbDictionary,noshowSpamProb

def Test(str,stopWordsPath):
    rawWordList = re.split(' ',str)
    wordList = [singleToken.lower() for singleToken in rawWordList if len(singleToken) > 1]
    if(stopWordsPath != None):
        f = open(stopWordsPath, 'r',encoding='iso-8859-1')
        readLines = f.readlines()
        stopWords =[token.strip() for token in readLines]
        
        for every_word in wordList:
              if(every_word in stopWords):
                wordList.remove(every_word)
#    print wordList.__len__(),"after"
    #print stopwords[100]
    return wordList

def Train(spam,dataPath,stopWordsPath,classifier):
    prior = fnPrior(dataPath,classifier)
    spamCount = prior[0]
    spamText = prior[1]
    spamWordsList = Test(spamText,stopWordsPath)
    spamList = spamConditionalProbablity(spamText,stopWordsPath)
    spamDataList = spamList[2]
    sum =0
    for i in spamList[2]:
        sum+= spamDataList[i]
    return spamCount,spamText,spamList[2],spamList[3],spamWordsList.__len__()

def TrainMultinomail(ham,hamDataPath,spam,spamDataPath,stopWordsPath):
    spamData = Train(spam,spamDataPath,stopWordsPath,"spam")
    hamData = Train(ham,hamDataPath,stopWordsPath,"ham")
    hamCount = hamData[4]
    spamCount = spamData[4]
    #print(hamCount + spamCount , "total words count")
    totalDocsCount = spamData[0] + hamData[0]
    global spamprior
    global hamprior
    spamprior= spamData[0]/totalDocsCount
    hamprior = hamData[0]/totalDocsCount
                      
    spamVocab = spamData[1]
    hamVocab = hamData[1]
    
    spamCondProb = spamData[2]
    hamCondprob = hamData[2]
    return hamVocab,spamVocab, hamprior,spamprior,hamCondprob,spamCondProb,hamData[3],spamData[3]


def hamTest(classifier,dataPath,hamTrainPath,spamTrainPath,stopWordsPath):
  global sectionHamTest
  global hamsuccesscount
  global hamfailurecount
  hamfailurecount=0
  sectionHamTest = " "
  hamsuccesscount = 0
  hamPath = dataPath
  hamdir = os.listdir(hamPath)
  hamTestCount=0
  parameters = TrainMultinomail("ham",hamTrainPath,"spam",spamTrainPath,stopWordsPath)
  for every_file in hamdir:
      hamTestCount+= 1
      hamFilesPath = hamPath  +'/'+every_file
      f = open(hamFilesPath,'r',encoding='iso-8859-1')
      sectionHamTest = f.read()
      tokens = Test(sectionHamTest,stopWordsPath)
      hamscore = log(parameters[2],10)
      spamscore = log(parameters[3],10)
      hamcondprob = parameters[4]
      spamcondprob = parameters[5]
      hamnoshowprob = parameters[6]
      spamnoshowprob = parameters[7]
      for  tokenkey in tokens:
          if tokenkey  in hamcondprob:
               hamscore+= log(hamcondprob[tokenkey],10)
          else:
               hamscore+= log(hamnoshowprob,10)
      for tokenkey in tokens:
          if  tokenkey in spamcondprob:
            spamscore += log(spamcondprob[tokenkey],10)
          else:
            spamscore += log(spamnoshowprob,10)

      if hamscore >= spamscore:
          hamsuccesscount+=1
      else:
          hamfailurecount+=1
  #print hamsuccesscount,hamfailurecount
  return hamsuccesscount,hamfailurecount

def spamTest(classifier,dataPath,hamTrainPath,spamTrainPath,stopWordsPath):

  global sectionSpamTest
  sectionSpamTest = " "
  global spamsuccesscount
  global spamfailurecount
  spamsuccesscount = 0
  spamfailurecount = 0
  spamPath = dataPath
  spamdir = os.listdir(spamPath)
  spamTestCount=0
  parameters = TrainMultinomail("ham",hamTrainPath,"spam",spamTrainPath,stopWordsPath)


  for every_file in spamdir:
      spamTestCount+= 1
      spamFilesPath = spamPath  +'/'+every_file
      f = open(spamFilesPath,'r',encoding='iso-8859-1')
      sectionSpamTest = f.read()
      #tokens = parseText(sectionSpamTest)
      tokens = Test(sectionSpamTest,stopWordsPath)
      hamscore = log(parameters[2],10)
      spamscore = log(parameters[3],10)
      hamcondprob = parameters[4]
      spamcondprob = parameters[5]
      hamnoshowprob = parameters[6]
      spamnoshowprob = parameters[7]
      for  tokenkey in tokens:
          if tokenkey in hamcondprob:
            hamscore+= log(hamcondprob[tokenkey],10)
          else:
               hamscore+= log(hamnoshowprob,10)
      for tokenkey in tokens:
          if tokenkey in spamcondprob:
            spamscore += log(spamcondprob[tokenkey],10)
          else:
            spamscore += log(spamnoshowprob,10)
      if hamscore <= spamscore:
          spamsuccesscount+=1
      else:
          spamfailurecount+=1
  #print spamsuccesscount,spamfailurecount
  return spamsuccesscount,spamfailurecount

def mainfn():
    if len(sys.argv) != 5:
        print("Invalid input arguments. Please specify input as : python finename.py trainhampath trainspampath testhampath testspampath stopwordspath")
        sys.exit(1)

    hamTrainPath = sys.argv[1]
    spamTrainPath = sys.argv[2]
    hamTestPath = sys.argv[3]
    spamTestPath = sys.argv[4]
    stopWordsPath= "stopwords.txt"
    ham = hamTest("", hamTestPath,hamTrainPath,spamTrainPath,stopWordsPath)
    spam = spamTest("", spamTestPath,hamTrainPath,spamTrainPath,stopWordsPath)
    print("Naive Bayes Classification without stop words:")
    print("Number of ham files successfully classified:",ham[0])
    print("Number of ham files successfully not classified: ",ham[1])
    print("Number of spam files successfully classified: ",spam[0])
    print("Number of spam files successfully not classified: ",spam[1])
    Numerator = ham[0] + spam[0]
    Denominator = ham[0] + ham[1]+ spam[0] +spam[1]
    Accuracy = (Numerator/Denominator)*100
    print("The total accuracy is: ",Accuracy)
    print('\n')
    stopWordsPath= None
    ham = hamTest("", hamTestPath,hamTrainPath,spamTrainPath,stopWordsPath)
    spam = spamTest("", spamTestPath,hamTrainPath,spamTrainPath,stopWordsPath)
    print("Naive Bayes Classification with stop words:")
    print("Number of ham files successfully classified:",ham[0])
    print("Number of ham files successfully not classified: ",ham[1])
    print("Number of spam files successfully classified: ",spam[0])
    print("Number of spam files successfully not classified: ",spam[1])
    Numerator = ham[0] + spam[0]
    Denominator = ham[0] + ham[1]+ spam[0] +spam[1]
    Accuracy = (Numerator/Denominator)*100
    print("The total accuracy is: ",Accuracy)
    
if __name__ == "__main__":
	mainfn()


