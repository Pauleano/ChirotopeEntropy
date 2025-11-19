import numpy as np
import itertools as iter
import time

def generate_matrix_from_string(input_str,emb_dim):
    """
    Diese Funktion berechnet die Lambda-Matrix aus 'Multidimensional Sorting' von Goodman und Pollack eines gegebene Chirotops (lexikografische Reihenfolge der Punkttripel).
    Einträge der Lambda-matrix entsprchen der anzahl der Pukte auf der positiven Seite eine Gerade.
    Die gerade verläuft durch zwei Punkte.
    Der Zeilenindex gibt den Punkt des ersten Punktes an und der Spaltenindex den Index des zweiten Punktes.

    Args:
    - input_str (string): Das Chirotop, eine Sequenz aus '+' und '-'.
    - emb_dim (int): Einbettungsdimension, bestimmt die Größe der quadratischen Matrix.
    
    Returns:
    - lambda_matrix (ndarray): Lambda-Matrix des Chirotops
    """

        # Create a string consisting of all indices
    zahlen = ''.join(str(m) for m in range(emb_dim))
    
    # Get all valid combinations of indices and convert them to integers
    iter_kombinations = list(iter.combinations(zahlen, 3))
    int_combinations = np.array([list(kombi) for kombi in iter_kombinations], dtype=int)
    #print(int_combinations)
    #initialisiere lambda-matrix 
    lambda_matrix = np.zeros((emb_dim, emb_dim), dtype=int)-np.eye(emb_dim)
    
    #iteriere über die zeichen des Chirotops und addiere entsprechend innerhalb der matrix
    for idx, comb in enumerate(int_combinations):
        char = input_str[idx]
        #print(char)
        if char == '+':
            lambda_matrix[comb[0], comb[1]] += 1
            lambda_matrix[comb[1], comb[2]] += 1
            lambda_matrix[comb[2], comb[0]] += 1
            
        if char== '-':
            lambda_matrix[comb[1], comb[0]] += 1
            lambda_matrix[comb[2], comb[1]] += 1
            lambda_matrix[comb[0], comb[2]] += 1
        #print(lambda_matrix)
    return lambda_matrix


def ordertype_check(chiro1_in,chiro2_in,order):
    '''
    Function which takes two chirotopes in "+-" encoding as input (lexikographical triples) and returns "True" or "False" depending on whether the two chirotopes belong to the same ordertype  
    '''
    chiro1=chiro1_in #auch weglassbar
    chiro2=chiro2_in
    
    iteration=1 #für abbruchkriterium
    while iteration<3:
        lambda_matrix_reference=generate_matrix_from_string(chiro1,order)
        #print(f"first lambda_matrix:\n{lambda_matrix_reference}")
        #get canonical ordering of first lambda-matrix (dont have to worry about negative entry, as negative entry is first point)
        #negative row because we need to sort in descending order
        sorted_indices = np.argsort(-lambda_matrix_reference[0,:])
        canonical_ordering_reference = np.arange(order)[sorted_indices]
        #shift last to element to first element to get correct canonical ordering
        canonical_ordering_reference=np.roll(canonical_ordering_reference, 1)
        #print(f"canonical ordering relative to first point:\n{canonical_ordering_reference}")

        lambda_matrix_iter=generate_matrix_from_string(chiro2,order)
        #print(f"lambda_matrix of chiro {chiro}:\n{lambda_matrix_iter}")
        #get extreme points
        #point k is extreme if there exists j, such that lambda(k,j)=0
        extreme_points = np.argwhere(lambda_matrix_iter == order-2)[:,0]
        #print(f"extreme points of chiro {chiro}:\n{extreme_points}")
        for k in extreme_points:
            #sort entries in k-th row, to get canonical ordering
            sorted_indices_iter = np.argsort(-lambda_matrix_iter[k,:])
            canonical_ordering_iter = np.arange(order)[sorted_indices_iter]
            #shift last to element to first element to get correct canonical ordering
            canonical_ordering_iter=np.roll(canonical_ordering_iter, 1)    
            #print(f"canonical ordering of extreme point {k}:\n{canonical_ordering_iter}")

            #apply permutation to tripels
            zahlen = ''.join(str(m) for m in range(order))
            iter_kombinations = list(iter.combinations(zahlen, 3))
            int_combinations = np.array([list(kombi) for kombi in iter_kombinations], dtype=int)
            #initialise permutated_combinations
            permutated_combinations=int_combinations
            #apply permutation for each entry to permutated_combinations
            for l in range(canonical_ordering_iter.shape[0]):
                permutated_combinations=np.where(int_combinations==canonical_ordering_reference[l],canonical_ordering_iter[l],permutated_combinations)
            #print(f"combinations after applying permutation:\n{permutated_combinations}")
            
            #get new lambda-matrix and compare with lambda-matrix_reference
            lambda_matrix_reference_permutated = np.zeros((order, order), dtype=int)-np.eye(order)
            for idx, comb in enumerate(permutated_combinations):
                char = chiro1[idx]
                if char == '+':
                    lambda_matrix_reference_permutated[comb[0], comb[1]] += 1
                    lambda_matrix_reference_permutated[comb[1], comb[2]] += 1
                    lambda_matrix_reference_permutated[comb[2], comb[0]] += 1
                if char== '-':
                    lambda_matrix_reference_permutated[comb[1], comb[0]] += 1
                    lambda_matrix_reference_permutated[comb[2], comb[1]] += 1
                    lambda_matrix_reference_permutated[comb[0], comb[2]] += 1
            #print(f"umsortierte lambda-matrix:\n{lambda_matrix_iter_permutated}")
            if (lambda_matrix_reference_permutated==lambda_matrix_iter).all():
                return True
        #if no relabeling is found, check inverse of chiro2     
        iteration+=1
        chiro2=chiro2.replace("+", "x").replace("-", "+").replace("x", "-")
    
    #if no relabeling of chiro1 is found that matches Lambda_Matrix of chiro2 or chiro2_inverse, then chiro1 belongs to a different ordertype than chiro2 
    return False


def ordertypes_from_chirotopes(input_list,order):
#input_list is a list of chirotopes (+- encoding in lexikografical)
#filter function:
#outputs two lists:
# list1 contains all chirotopes of the same ordertype as first element of input_list
# list2 contains all chirotopes which belong to a different ordertype   
#function looks for a relabeling of points of first chiro to get the lambda-matrices of first chiro and other chiro to match
    same_ordertype_list=[]
    same_ordertype_list.append(input_list[0])
    different_ordertype_list=[]

    for iter_chiro in input_list[1:]:
        same_OT=ordertype_check(input_list[0],iter_chiro,order)
        if (same_OT):
            same_ordertype_list.append(iter_chiro)
        else:
            different_ordertype_list.append(iter_chiro)
                
    return same_ordertype_list,different_ordertype_list


def index_of_ordertypes(chirotope_input,emb_dim):
    """
    Diese Funktion gibt die Indizes der Chirotope eines Ordnungstyps an.
    Methode funktioniert nur bis m=5 da sonst nicht mehr  #Ordnungstype=emb_dim-2 gilt
    Args:
    - chirotope_input (ndarray): Array dessen Einträge aus Chirotopen bestehen, welche anhand '+' und '-' dargstellt sind.
    - emb_dim (int): Einbettungsdimension, bestimmt die Anzahl der Ordnungstype
    Returns:
    -unique: the different number of extreme points
    -counts: the frequency of chirotopes with each number of extreme points
    -index_lists: a list for each number of extreme points with the indexes of the chirotope
    """

    extrem_points_count_list=[]
    for input_string in chirotope_input:
        #print(input_string)
        matrix = generate_matrix_from_string(input_string,emb_dim)
        #print(matrix)
        positions = np.argwhere(matrix == emb_dim-2)[:,0]
        #print(positions)
        extrem_points_count_list.append(len(positions))
    #print(extrem_points_count_list)    
    unique,counts=np.unique(extrem_points_count_list,return_counts=True)
    index_lists=[]
    for k in range(len(unique)):
        index_lists.append([i for i, val in enumerate(extrem_points_count_list) if val == unique[k]])
        
    return unique,counts,index_lists
 

def list_of_ordertypes(chiro_input,index_input):
   
    order_arr=np.array([1,4,10,20,35])
    order=np.argwhere(order_arr==len(chiro_input[0])).item()+3
    #print(order)
    
    ordertype_lists=[]
    index_lists=[]
    iter_chiros=chiro_input
    iter_indizes=index_input
    #print(len(iter_chiros))
    while len(iter_chiros)!=0:
        print(len(iter_chiros))
        ordertype_list,remaining_chiros=ordertypes_from_chirotopes(iter_chiros,order)
        #print(ordertype_list)
        ordertype_lists.append(np.array(ordertype_list))
        mask=np.isin(iter_chiros,ordertype_list)
        #print(mask)
        index_lists.append(remaining_chiros)
        iter_chiros=iter_chiros[~mask]
        iter_indizes=iter_indizes[~mask]

    return ordertype_lists,index_lists

