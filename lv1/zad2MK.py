ocjena = float(input("Unesi ocjenu"))


if ocjena > 1.0 or ocjena < 0.0:
    raise Exception('Nije u rangu')

if ocjena >= 0.9:
    print ('A')
elif ocjena >= 0.8:
    print ('B')
elif ocjena >= 0.7:
    print ('C')
elif ocjena >= 0.6:
    print ('D')
elif ocjena < 0.6:
    print ('F')    

