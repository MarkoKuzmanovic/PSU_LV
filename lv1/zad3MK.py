loop = True
lista = []
brojPetlji = 0

while loop:
    
    broj = input()
    if broj == 'Done':
        loop = False
        break
    try:
        broj = float(broj)
    except:
        print ('nije broj!')
        continue    
    lista.append(broj)
    brojPetlji += 1

print('Korisnik je unio: ' + str(brojPetlji) + ' brojeva')
print (lista)
avg = sum(lista) / len(lista)
print ('Srednja vrijednost liste: ' + str(avg) )
print ('Max: ' + str(max(lista)))
print ('Min ' + str(min(lista)))
lista.sort()
print (lista)


        
