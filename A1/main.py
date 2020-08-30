import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
	# input:pandas_dataframe
	# output:int/float/double/large


def get_entropy_of_dataset(df):
	entropy = 0
	return entropy


def entropyFormula():

'''Return entropy of the attribute provided as parameter'''
	# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	# output:int/float/double/large


def get_entropy_of_attribute(df, attribute):
	entropy_of_attribute = 0

  # Steps to solve this function
  # 0. Get the list of values the attributes can take
  valuesOfTheAttribute = df[attribute].unique().tolist()
  
  valueOfTheAttribute_Dataframe = {}
  
  # 1. Create separate dataframes view for each possbile value of the attribute
  for value in valuesOfTheAttribute:
    valueOfTheAttribute_Dataframe[value] = df[df.attribute == value]
    
	# 2. For each dataframe
  for value_dataframe in valueOfTheAttribute_Dataframe:
    # 2.1 Find to the number of 'Yes' and 'No' entries in the dataframe
    noOfYes, noOfNo = value_dataframe[1][df.play == "Yes"].count(), value_dataframe[1][df.play == "No"].count()
    # 2.2 Use the entropy formula and get the entropy of the attribute value pair
    
    # 2.3 Add the results to the running sum
  # 3. Return the sum
  
  	
	return abs(entropy_of_attribute)




def get_information_gain(df,attribute):
	return abs(get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute))


def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''
	max_col = float('-inf')
  
	cols = df.columns
	for i in range(len(cols)):
		information_gains[cols[i]] = get_information_gain(df, cols[i])
		if(max_col < information_gains[cols[i]]):
			max_col = information_gains[cols[i]]
			selected_column = cols[i]

	return (information_gains,selected_column)