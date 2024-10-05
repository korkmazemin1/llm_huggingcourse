# import necessary libraries
from transformers import pipeline
import pandas as pd 

# we choose the text classficiaton from pipeline 
classifier = pipeline("text-classification")

# here input text
text= " we love breakfast"

# we can use the base modek with text then take the outputs
outputs= classifier(text)

# convert outputs a dataframe 
pd.DataFrame(outputs)

