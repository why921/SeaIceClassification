'''
ALPSRP180031440
ALPSRP233871400
ALPSRP258351570
ALPSRP258351550
ALPSRP205991520
ALPSRP171831460
'''
import glob

pauli=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_24.txt")
spectrogram=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_spe_24_4bands.txt")


c1 = open("pauliDataPath.txt", 'w').close()
c2 = open("spectrogramDataPath.txt", 'w').close()

with open("pauliDataPath.txt", "a") as file1:
    for i in range(0, len(pauli)):
        f1=open(pauli[i]).read()
        file1.write(f1)
with open("spectrogramDataPath.txt", "a") as file2:
    for j in range(0, len(spectrogram)):
        f2=open(pauli[j]).read()
        file2.write(f2)