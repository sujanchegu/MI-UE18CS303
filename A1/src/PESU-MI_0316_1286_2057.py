import numpy as np
import pandas as pd
import random


def get_entropy_of_dataset(df):
    # gets the target column
    target = df.loc[:, df.columns[-1]]
    unique_values = target.unique().tolist()

    # number of unique values
    uniq = len(unique_values)

    target = target.tolist()
    vals = []
    for i in range(uniq):
        vals.append(target.count(unique_values[i]))
    summ = sum(vals)
    entropy = 0
    for i in range(uniq):
        entropy += (-1)*(vals[i]/summ)*(np.log2(vals[i]/summ))
    return entropy


def entropyFormula(answerDict):
    valueOfAttribute_entropy = {}

    for value in answerDict:
        temp = answerDict[value].values()
        denominator = sum(temp)
        entropy = 0

        for count in temp:
            if (count != 0):
                entropy += -(count/denominator) * np.log2(count/denominator)

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


def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0

    if attribute not in df.columns.tolist():
        return entropy_of_attribute

    # Get the name of the target attribute column
    TARGET_ATTRIBUTE = df.columns.tolist()[-1]

    # Get the list of unique values the attribute can take
    valuesOfAttribute = df[attribute].unique().tolist()

    # Get the list of unique values the TARGET_ATTRIBUTE can take
    valuesOfTargetAttribute = df[TARGET_ATTRIBUTE].unique().tolist()

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

    # print(answerDict)

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

    # print(answerDict)

    # Use the entropy formula and get the entropy of all the attribute
    # value pairs and take the sum
    valueOfAttribute_entropy = entropyFormula(answerDict)
    entropy_of_attribute = avgInformationEntropy(answerDict,
                                                 valueOfAttribute_entropy,
                                                 df.shape[0])

    # 3. Return the sum
    return abs(entropy_of_attribute)


def get_information_gain(df, attribute):
    return abs(get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute))


def get_selected_attribute(df):
    information_gains = {}
    selected_column = ''
    max_col = float('-inf')

    cols = df.columns
    for i in range(len(cols)-1):
        information_gains[cols[i]] = get_information_gain(df, cols[i])
        if(max_col < information_gains[cols[i]]):
            max_col = information_gains[cols[i]]
            selected_column = cols[i]

    return (information_gains, selected_column)
