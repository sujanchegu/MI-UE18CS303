#SentinelPrime7 here
'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

# import math

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	return entropy


def entropyFormula():
  


'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
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



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0
  #Steps:
  #0. Given the attribute for which IG should be calculated
  #1. 
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''

'''
In the function get_information_gain(), the description given is:
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute)

The input types might be wrong. The parameter is given as df b