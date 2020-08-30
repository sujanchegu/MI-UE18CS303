import numpy as np
import pandas as pd
import random
import math


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float/double/large


def get_entropy_of_dataset(df):
    entropy = 0
    return entropy


def entropyFormula(answerDict):
    valueOfAttribute_entropy = {}

    for value in answerDict:
        temp = answerDict[value].values()
        denominator = sum(temp)
        entropy = 0

        for count in temp:
            if (count != 0):
                entropy += -(count/denominator) * math.log2(count/denominator)

        valueOfAttribute_entropy[value] = entropy

    return valueOfAttribute_entropy


def avgInformationEntropy(answerDict, valueOfAttribute_entropy,
                          totalNumberOfSamples):
    answer = 0
    for valueOfAttribute in answerDict:
        numerator = sum(answerDict[valueOfAttribute].values())
        answer += numerator / totalNumberOfSamples * \
            valueOfAttribute_entropy[valueOfAttribute]

    return answer


'''Return entropy of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large


def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0

    if attribute not in df.columns:
        return entropy_of_attribute

    # Get the name of the target attribute column
    TARGET_ATTRIBUTE = df.columns[-1]

    # Get the list of unique values the attribute can take
    valuesOfAttribute = df[attribute].unique().tolist()

    # Get the list of unique values the TARGET_ATTRIBUTE can take
    valuesOfTargetAttribute = df[TARGET_ATTRIBUTE].unique().to_list()

    # Create a dictionary containing the values of attribute as keys
    # and the values as the outcome:no_of_occurrences
    # E.g.:
    # {
    # 	'val1': {'a':1, 'b':0},
    # 	'val2': {'a':0, 'b':2}
    # }
    answerDict = {}
    for value in valuesOfAttribute:
        answerDict[value] = dict.fromkeys(valuesOfTargetAttribute, 0)

    # Create a dictionary to hold the unique attribute value with the
    # corresponding df as key-value pairs
    valuesOfAttribute_Dataframe = {}

    # Fill in the values in to the: valuesOfAttribute_Dataframe, dictionary
    for value in valuesOfAttribute:
        valuesOfAttribute_Dataframe[value] = df.loc[df[attribute] == value]

    # For each dataframe
    for valueOfAttribute in valuesOfAttribute_Dataframe:
        dataframe = valuesOfAttribute_Dataframe[valueOfAttribute]

        for valueOfTargetAttribute in valuesOfTargetAttribute:
            # Find to the count of different entries in the target attribute
            temp = dataframe.loc[dataframe[TARGET_ATTRIBUTE] ==
                                 valueOfTargetAttribute] \
                                 .count().tolist()[-1]

            answerDict[valueOfAttribute][valueOfTargetAttribute] = temp

    # Use the entropy formula and get the entropy of all the attribute
    # value pairs and take the sum
    valueOfAttribute_entropy = entropyFormula(answerDict)
    entropy_of_attribute = avgInformationEntropy(answerDict,
                                                 valueOfAttribute_entropy,
                                                 df.shape()[0])

    # 3. Return the sum
    return abs(entropy_of_attribute)




def get_information_gain(df,attribute):
    return abs(get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute))


def get_selected_attribute(df):
    information_gains = {}
    selected_column = ''
    max_col = float('-inf')

    cols = df.columns
    for i in range(len(cols)):
        information_gains[cols[i]] = get_information_gain(df, cols[i])
        if(max_col < information_gains[cols[i]]):
            max_col = information_gains[cols[i]]
            selected_column = cols[i]

    return (information_gains, selected_column)
