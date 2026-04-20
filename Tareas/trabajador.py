def evaluar_lote(lote):
    suma = 0.0
    val_min = float('inf')
    val_max = float('-inf')
    
    for x in lote:
        suma += x
        if x < val_min: 
            val_min = x
        if x > val_max: 
            val_max = x
            
    media = suma / len(lote)
    return media, val_min, val_max