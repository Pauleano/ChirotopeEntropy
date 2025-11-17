from ordertype_funcs import list_of_ordertypes,index_of_ordertypes,ordertypes_from_chirotopes
import numpy as np

#testing example
filename=f"chirotopes_uniform_order_{5}_{100000}_Punkte.txt"
chirotopes = np.loadtxt(filename,dtype=str)
unique,counts,indexe=index_of_ordertypes(chirotopes,5)
print(unique)
print(counts)
for i in indexe:
    print(i)


#testing example
chiro_list=[
    '+++++++++++-------++','++++++++++++-------+','------------+++++++-']   
same,different=ordertypes_from_chirotopes(chiro_list,6)
print(f"same1:{same}")
print(different)

#testing example
filename=f"chirotopes_uniform_order_{6}_{4000000}_Punkte.txt"
chirotope_data = np.loadtxt(filename,dtype=str)

chiro_input=chirotope_data[0:1]
ordertypes,index_lists=list_of_ordertypes(chiro_input,np.arange(chiro_input.shape[0]))
#filename=f"indices_ordertypes_6_Punkte_lexiko.txt"
for i in range(len(index_lists)):
    print(ordertypes[i])
print(len(index_lists))
    #with open(filename, "a") as file:
        #line = " ".join(str(x) for x in i)
        #file.write(line + "\n")

for i in ordertypes:
    a,b,_=index_of_ordertypes(i,6)
    print(a)
    print(b)
    
