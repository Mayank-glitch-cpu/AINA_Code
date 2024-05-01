import json
import time

st= time.time()

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Node of a Huffman Tree  
class Nodes:  
    def __init__(self, probability, symbol, left = None, right = None):  
        # probability of the symbol  
        self.probability = probability  
  
        # the symbol  
        self.symbol = symbol  
  
        # the left node  
        self.left = left  
  
        # the right node  
        self.right = right  
  
        # the tree direction (0 or 1)  
        self.code = ''  
  
""" A supporting function in order to calculate the probabilities of symbols in specified data """  
def CalculateProbability(the_data):  
    the_symbols = dict()  
    for item in the_data:  
        if the_symbols.get(item) == None:  
            the_symbols[item] = 1  
        else:   
            the_symbols[item] += 1       
    return the_symbols  
  
""" A supporting function in order to print the codes of symbols by travelling a Huffman Tree """  
the_codes = dict()
def clear_dict():
    the_codes.clear()  
  
def CalculateCodes(node, value = ''):  
    # a huffman code for current node  
    newValue = value + str(node.code)  
  
    if(node.left):  
        CalculateCodes(node.left, newValue)  
    if(node.right):  
        CalculateCodes(node.right, newValue)  
  
    if(not node.left and not node.right):  
        the_codes[node.symbol] = newValue  
           
    return the_codes  
  
""" A supporting function in order to get the encoded result """  
def OutputEncoded(the_data, coding):  
    encodingOutput = []  
    for element in the_data:  
        # print(coding[element], end = '')  
        encodingOutput.append(coding[element])  
          
    the_string = ''.join([str(item) for item in encodingOutput])      
    return the_string  
          
""" A supporting function in order to calculate the space difference between compressed and non compressed data"""      
def TotalGain(the_data, coding):  
    # total bit space to store the data before compression  
    beforeCompression = len(the_data) * 8  
    afterCompression = 0  
    the_symbols = coding.keys()  
    for symbol in the_symbols:  
        the_count = the_data.count(symbol)  
        # calculating how many bit is required for that symbol in total  
        afterCompression += the_count * len(coding[symbol])  
    #print("Space usage before compression (in bits):", beforeCompression)  
    #print("Space usage after compression (in bits):",  afterCompression)  
  
def HuffmanEncoding(the_data):  
    symbolWithProbs = CalculateProbability(the_data)  
    the_symbols = symbolWithProbs.keys()  
    the_probabilities = symbolWithProbs.values()  
    #print("symbols: ", the_symbols)  
    #print("probabilities: ", the_probabilities)  
      
    the_nodes = []  
    #encoded=[]  
    # converting symbols and probabilities into huffman tree nodes  
    for symbol in the_symbols:  
        the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))  
    size=0  
    while len(the_nodes) > 1:  
        # sorting all the nodes in ascending order based on their probability  
        the_nodes = sorted(the_nodes, key = lambda x: x.probability)  
        # for node in nodes:    
        #      print(node.symbol, node.prob)  
      
        # picking two smallest nodes  
        right = the_nodes[0]  
        left = the_nodes[1]  
      
        left.code = 0  
        right.code = 1  
      
        # combining the 2 smallest nodes to create new node  
        newNode = Nodes(left.probability + right.probability, left.symbol + right.symbol, left, right)  
      
        the_nodes.remove(left)  
        the_nodes.remove(right)  
        the_nodes.append(newNode)
        size+=1  
    huffmanEncoding = CalculateCodes(the_nodes[0])
    #print("symbols with codes", huffmanEncoding)  
    TotalGain(the_data, huffmanEncoding)  
    encodedOutput = OutputEncoded(the_data,huffmanEncoding)
    #encoded.append(encodedOutput)
    #print('Size of tree= ',size)
    #temp=dict()
    #temp.clear()
    #temp=huffmanEncoding
    #huffmanEncoding.clear()
    return encodedOutput, huffmanEncoding  

def HuffmanDecoding(encodedData, huffmanTree):  
    treeHead = huffmanTree
    decodedOutput = []
    try_string=''
    key_list= list(treeHead.keys())
    val_list=list(treeHead.values())
    #print(key_list)
    #print(val_list)
    for x in encodedData:
        try_string= try_string+x
        #print(int(try_string))  
        if try_string in  val_list :
            position= val_list.index(try_string)  
            decodedOutput.append(key_list[position])     
            try_string=''
    return decodedOutput



def flatten(xss):
    return [x for xs in xss for x in xs]
#The Following function creates a dataset that reads a csv file and precisely read the columns of temperature and humidity.
#and store it in the 2d array 'X'.
dataset = pd.read_csv('Wheat_input.csv')
X= dataset.iloc[:,[4,5]].values
print (X)
no_clusters=5


#The later code is making clusters of whole data using "AGGLOMERATIVE Heirarchical CLUSTERING " technique basically this technique deals with 
# clustering on bottom to top basis, like initially all the clusters will be equal to number of data points, and then it is mergerd using ward 
#linkage technique.
#the 'y_hc' variable preduct the data that when a new data comes into which cluster it would go based on the traing it has recieved 
hc= AgglomerativeClustering(n_clusters= no_clusters , affinity= 'euclidean', linkage='ward')
y_hc= hc.fit_predict(X)

cluster_map= pd.DataFrame(dataset)
cluster_map['data_index']= dataset.index.values
cluster_map['cluster']= hc.labels_
#cluster_map.sort_values(by=['data_index'], ascending=False)

#The following function displays the whole dataframe along with the cluster the data belong to.
#print(cluster_map)
with open("F:\IIIT_kancheepuram_Internship_project\Data Compression\data_visualisation\wheat_tree.txt",'w') as f1:
    with open("F:\IIIT_kancheepuram_Internship_project\Data Compression\data_visualisation\wheat_encoded.txt",'w') as f2:
    #the below forms different clusters and 
        clusters=[]
        for k in range(no_clusters):
           clusters.append(cluster_map[cluster_map.cluster ==k])
        #print(clusters)   
        # it will form two text files one for tree and for encoded output 
        #print(len(dataset.columns))
        #print(no_clusters)
        for n in range(len(dataset.columns)-1):
            for k in range(no_clusters):
                display_data=clusters[k].iloc[:,n].tolist()
                #print(display_data)
                #print('******************************cluster',k,dataset.columns[n],'*************************************************')
                #print(display_data)
                #display(display_data)
                if n>=2:
                    int_data=[]
                    for datak in display_data:
                        int_data.append(round(float(datak)))
                            
                    encoding, tree= HuffmanEncoding(int_data)
                    #print(tree)
                    f2.write(encoding)
                    f2.write("\n")
                    f1.write(json.dumps(tree))
                    f1.write('\n')   
                    #f1.write(json.dumps(tree))
                    #f1.write('\n')
                    #print("Decoded Output ",dataset.columns[n]," data",HuffmanDecoding(encoding,tree))
                    
                    
                    #print(int_data)
                else:
                      
                    #print(str_data)    
                    encoding, tree= HuffmanEncoding(display_data)
                    #print(tree)
                    f2.write(encoding)
                    f2.writelines("\n")
                    f1.write(json.dumps(tree))
                    f1.write('\n')  
                    #f1.write(json.dumps(tree))
                    #f1.write('\n')
                    #print("Encoded Output ",dataset.columns[n]," data", encoding) 
                    
                    #print("Decoded Output ",dataset.columns[n]," data",HuffmanDecoding(encoding,tree))
                clear_dict()    
        
et= time.time()
elapsed_time2= et - st
print('Elapsed Time', elapsed_time2, 'secs')              