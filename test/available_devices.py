import sounddevice as sd
import numpy as np

print(sd.query_devices())
fs = 16000
duration = 3
sd.default.samplerate = fs
sd.default.channels = 1
print("Felvétel indul...")
felvetel = sd.rec(int(duration * fs))
sd.wait()
print("Lejátszás indul...")
sd.play(felvetel, fs)   
sd.wait()
print("Kész.")
