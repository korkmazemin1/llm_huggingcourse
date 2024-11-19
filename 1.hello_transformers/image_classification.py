from PIL import Image 
import requests
from  transformers import pipeline

url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg"
# webde resmi çekmek için url 


image=Image.open(requests.get(url,stream=True).raw)
# webe istek atarak url ile resmi koda çekiyoruz

classifier= pipeline("image-classification")
# transformers kütüphanesi içerisinde sınıflayıcı çağırdım

outputs=classifier(image)
# sınflandırma işlemini gerçekleştirip sonuçları yazdırıyoruz
print(outputs)

# kursta birkaç örnek daha var ancak bu kadarı yeterli zaten görüntü işleme ile yeterince uğraştım 



