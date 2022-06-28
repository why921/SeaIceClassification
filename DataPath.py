import numpy as np
import glob

pauli=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_SLC_36.txt")
#spectrogram=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_spe_24_4bands.txt")

c1 = open("E:\ALOSPALSAR\TrainData\SLC36.txt", 'w').close()


with open("E:\ALOSPALSAR\TrainData\SLC36.txt", "a") as file1:
    for i in range(0, len(pauli)):
        f1=open(pauli[i]).read()
        file1.write(f1)


