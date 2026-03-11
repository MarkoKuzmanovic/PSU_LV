trazeniString = ('X-DSPAM-Confidence:')
Pouzdanosti = []
imeDatoteke = input()
Datoteka = open('C:/Users/student/Desktop/lv1/' + imeDatoteke)

lines = Datoteka.readlines()
for line in lines:
    if line.startswith(trazeniString):
        data = line.split()[-1]
        Pouzdanosti.append(float(data))    

avg = sum(Pouzdanosti) / len(Pouzdanosti)
print (avg)
Datoteka.close()

