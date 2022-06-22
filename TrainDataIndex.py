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

pauli36=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_36.txt")
spectrogram36=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_spe_36_4bands.txt")

pauli48=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_48.txt")
spectrogram48=glob.glob(r"E:\ALOSPALSAR\TrainData\*\ALPSRP?????????_spe_48_4bands.txt")

c1 = open("pauliDataPath.txt", 'w').close()
c2 = open("spectrogramDataPath.txt", 'w').close()

c361 = open("pauliDataPath36.txt", 'w').close()
c362 = open("spectrogramDataPath36.txt", 'w').close()

c481 = open("pauliDataPath48.txt", 'w').close()
c482 = open("spectrogramDataPath48.txt", 'w').close()

with open("pauliDataPath.txt", "a") as file1:
    for i in range(0, len(pauli)):
        f1=open(pauli[i]).read()
        file1.write(f1)
with open("spectrogramDataPath.txt", "a") as file2:
    for j in range(0, len(spectrogram)):
        f2=open(spectrogram[j]).read()
        file2.write(f2)

with open("pauliDataPath36.txt", "a") as file1:
    for i in range(0, len(pauli36)):
        f1=open(pauli36[i]).read()
        file1.write(f1)
with open("spectrogramDataPath36.txt", "a") as file2:
    for j in range(0, len(spectrogram36)):
        f2=open(spectrogram36[j]).read()
        file2.write(f2)

with open("pauliDataPath48.txt", "a") as file1:
    for i in range(0, len(pauli48)):
        f1=open(pauli48[i]).read()
        file1.write(f1)
with open("spectrogramDataPath48.txt", "a") as file2:
    for j in range(0, len(spectrogram48)):
        f2=open(spectrogram48[j]).read()
        file2.write(f2)