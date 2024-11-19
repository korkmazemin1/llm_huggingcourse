from transformers import pipeline
import soundfile as sf
import numpy as np

text = "hello how are u "

# Text-to-speech modeli yüklendi
synth = pipeline("text-to-speech", model="suno/bark")

# Text-to-speech işlemi
speech = synth(text)

# Çıktıyı kontrol et ve gerekirse boyutları düzelt
audio_data = np.squeeze(speech["audio"])

# Ses dosyasını kaydet
sf.write("speech1.wav", audio_data, samplerate=speech["sampling_rate"])
# model ağır olduğu için çıktılar kasada 