import time
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
###########################################################
#generate chirotopes (lexikografical ordering)
###########################################################

def orientations_all_combinations_lexikografical(multivar_data, emb_dim, emb_delay):
    """
    Diese Funktion berechnet die Chirotope der Ausschnitte einer Zeitreihe.
    Die Ausschnitte sind anhand emb_dim und emb_delay definiert.
    Die Punkttripel werden in lexikografischer Ordungs durchlaufen.
    
    Args:
    - multivar_data (ndarray): 2D numpy array, jeweils eine zeile entspricht einer variable der betrachteten zweidimensionalen Zeitreihe.
    - emb_dim (int): Einbettungsdimension, bestimmt die Anzahl der Punkte in einem Ausschnitt.
    - emb_delay (int): Verzögerung/Abstand zwischen Datenpunkten eines Ausschnitts.
    
    Returns:
    - unique_symbols (ndarray): Die enzigartigen Chirotope von 'multivar_data'.
    - symbol_count (ndarray): Die Anzahl der jeweiligen Chirotope.
    """
    
    #definition eine string bestehend aus den indizes eines Ausschnitts
    zahlen = ''.join(str(m) for m in range(emb_dim))
    
    #generierung aller relevanten indizetripel und speicherung als liste
    iter_kombinations = list(iter.combinations(zahlen, multivar_data.shape[0] + 1))
    int_combinations = np.array([list(kombi) for kombi in iter_kombinations], dtype=int)

    #initialisierung der Matrix zur orientierungsberechnung
    matrix = np.concatenate((np.ones((multivar_data.shape[0] + 1, 1)), 
                             np.zeros((multivar_data.shape[0] + 1, multivar_data.shape[0]))), axis=1)
    
    # Liste der beobachteten Chirotope
    words = []

    for i in range(multivar_data.shape[1] - (emb_dim - 1) * emb_delay):
        word = []

        #betrachte relevanten punkte basierend auf emb_dim und emb_delay
        relev_points = multivar_data[:, i:i + emb_dim * emb_delay:emb_delay]
        
        #berechnung aller orientierungen
        for k in range(int_combinations.shape[0]):
            for l in range(int_combinations.shape[1]):
                #aussuchen des relvanten punkttripel
                matrix[l, 1:] = relev_points[:, int_combinations[k, l]]
            
            # Calculate the orientation by computing the sign of the determinant
            #print(matrix)
            orientation = np.sign(np.linalg.det(matrix))
            
            # Append symbol ('+', '-', or '0') based on orientation
            if orientation == 1:
                word.append('+')
            elif orientation == -1:
                word.append('-')
            else:
                word.append('0')
        
        #Hinzufügen zur chirotopliste
        words.append(''.join(word))

    #Ausgabe der einzigartigen chirotope and ihre häufigkeiten
    unique_symbols, symbol_count= np.unique(words, return_counts=True)
    return unique_symbols, symbol_count

#seed=249180981693180323229352415380346423354 #gotten by secrets.randbits(128)

#parameter
emb_dim_rauschen=5
emb_delay_rauschen=1
iterations_rauschen=10
log_base=2
mean=[0,0]
var_1=1
var_2=1
kov_12=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]

len_data=[200000]


for i in len_data:
    for j in kov_12:
        kovar=[[var_1, j],[j, var_2]]
        print(kovar)
        for k in range(iterations_rauschen):
        #zeitreihe_uniform=np.random.default_rng(seed+k).random((2,i))
            zeitreihe_normal=np.random.default_rng().multivariate_normal(mean,kovar,i).T

            start=time.time()
            chirotopes, freqs=orientations_all_combinations_lexikografical(zeitreihe_normal,emb_delay=emb_delay_rauschen,emb_dim=emb_dim_rauschen)
            print(freqs.shape)
            print('auswertung took %s sec.' %(time.time()-start))
            
            filename=f"chirotope_freqs_normal_order_{emb_dim_rauschen}_{i}_Punkte_var1_{var_1}_var2_{var_2}_kovar_{j}lexiko.txt"
            with open(filename, "a") as file:
                line = " ".join(str(x) for x in freqs)
                file.write(line + "\n")
        