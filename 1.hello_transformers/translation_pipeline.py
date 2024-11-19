# import necessary libraries
from transformers import pipeline
import pandas as pd 
from transformers import set_seed

text="hello my name is emin. ı am studying computer sciences in sakarya"

set_seed(0) # farklı denemelerde aynı sonucu vermesi için seed 0 a ayarlanır 

translator=pipeline("translation_en_to_de") # pipeline ile hangi dilden çevrim yapılcağı belirtir ve model yüklenir ü
# burada model parametresinde huggingfaceden spesifik bir model belirterek huggingfaceden herhangi bir moddel çekilebilir

outputs=translator(text)

print(outputs)
