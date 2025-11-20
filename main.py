from ordertype_funcs import list_of_ordertypes,index_of_ordertypes,ordertypes_from_chirotopes,entropies
import numpy as np
import time
import matplotlib.pyplot as plt

#testing index_of_ordertypes
filename=f"chirotopes_uniform_order_{5}_{100000}_Punkte.txt"
chirotopes = np.loadtxt(filename,dtype=str)
unique,counts,indexe=index_of_ordertypes(chirotopes,5)
print(unique)
print(counts)
for i in indexe:
    print(i)


#testing ordertypes_from_cirotopes
chiro_list=[
    '+++++++++++-------++','++++++++++++-------+','------------+++++++-']   
same,different=ordertypes_from_chirotopes(chiro_list,6)
print(f"same1:{same}")
print(different)


#testing list_of_ordertypes
filename=f"chirotopes_uniform_order_{5}_{100000}_Punkte.txt"
chirotope_data = np.loadtxt(filename,dtype=str)

chiro_input=chirotope_data[0:]
t=time.time()
ordertypes,index_lists=list_of_ordertypes(chiro_input,np.arange(chiro_input.shape[0]))

filename=f"ordertype_indizes_order_{5}_lexico.txt"
for i in index_lists:
    with open(filename, "a") as file:
        line = " ".join(str(x) for x in i)
        file.write(line + "\n")

for i in ordertypes:
    a,b,_=index_of_ordertypes(i,5)
    print(a)
    print(b)


#testing entropies-function
freqs=np.zeros((264,1))+1
chiros,ordertype,inside_ordertype=entropies(freqs,5)
print(chiros)
print(ordertype)
print(inside_ordertype)


#plotting
var_1=1
var_2=1
cov_12=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]

for i in cov_12:
    filename=f"chirotope_freqs_normal_order_{5}_{200000}_Punkte_var1_{var_1}_var2_{var_2}_kovar_{i}lexiko.txt"
    #filename=f"chirotope_freqs_normal_order_{4}_{50000}_Punkte_var1_{var_1}_var2_{var_2}_kovar_{i}lexiko.txt"
    chirotope_data = np.loadtxt(filename)
    for j in range(chirotope_data.shape[0]):
        plt.stairs(chirotope_data[j,:])
    mean_chirotope_data=np.mean(chirotope_data,axis=0)
    print(mean_chirotope_data.shape)
    over_chiros,over_ordertypes,within_ordertypes=entropies(mean_chirotope_data,5)
    print(f"entropy over all chiros: {over_chiros}")
    print(f"entropy over all ordeertypes: {over_ordertypes}")
    print(f"entropy within ordertypes:\n{within_ordertypes}")
    plt.show()

