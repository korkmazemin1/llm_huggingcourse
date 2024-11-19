from PIL import Image 
import requests
from  transformers import pipeline

url="https://i.pinimg.com/474x/56/5d/18/565d1841748d24275f5707d6c7043079.jpg"
# webde resmi çekmek için url 


image=Image.open(requests.get(url,stream=True).raw)
# webe istek atarak url ile resmi koda çekiyoruz

segmentor= pipeline("image-segmentation",model="mattmdjaga/segformer_b2_clothes")
# segmentasyon için transform çekilir 
outputs=segmentor(image)
# segmentasyon yapılır 

print(outputs)

mask=outputs[1]['mask']

image.show(mask)