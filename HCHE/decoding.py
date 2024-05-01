import ast
from Final_hoffmann import*
import csv

def flatten(xss):
    return [x for xs in xss for x in xs]
final_array=[]
try_array=[]
try_array2=[]
date=[]
time=[]
sm1=[]
sm2=[]
temp=[]
hum=[]
l=[]
p=[]
data_index=[]
#print(len(dataset.columns)*no_clusters)
with open('F:\IIIT_kancheepuram_Internship_project\Data Compression\data_visualisation\wheat_tree.txt','r') as f_txt1:
    r_=f_txt1.readlines()
#print(len(r_))



with open ('F:\IIIT_kancheepuram_Internship_project\Data Compression\data_visualisation\wheat_encoded.txt','r') as f_txt:
    encd=f_txt.readlines()
    #print(len(encd))

header= ['date','Time','Soil_moisture1','Soil_moisture2','Temp','Humidity','P','L',"Data_index"]
with open('F:\IIIT_kancheepuram_Internship_project\Data Compression\data_visualisation\Wheat_decoded.csv','w',newline='') as f_csv:
      writer=csv.writer(f_csv)
      writer.writerow(header)
#the outer "for" loop records for the number of columns in the dataframe as their are 7 columns exclusive
      k=1
      for n in range((len(dataset.columns)-1)*no_clusters):
          #the inner loop sync the cloumn data with the respective cluster
          # if n 
           display_data_=encd[n]
           #print(display_data_)
           dic= r_[n]
           tree__= ast.literal_eval(dic)
           #HuffmanDecoding(display_data_,tree__)
           
           try_array.append(HuffmanDecoding(display_data_,tree__))
           #print(int_data)
           
           if k%no_clusters==0:
               try_array2=flatten(try_array)
               final_array.append(try_array2)
               try_array=[]
               #try_array2=[]
               k=0
              
           k=k+1
           #print(k)
           #print(final_array)    
      #print(len(final_array))
    
    
      for n in range (9):  
         if n==0:
           date.append(final_array[0])
           f_date=flatten(date)                   
         if n==1:
             time.append(final_array[1])
             f_time= flatten(time)
         if n==2:
             sm1.append(final_array[2])
             f_sm1= flatten(sm1)
         if n==3:
             sm2.append(final_array[3])
             f_sm2= flatten(sm2)
         if n==4:
             temp.append(final_array[4])
             f_temp= flatten(temp)
         if n==5:
             hum.append(final_array[5])
             f_hum= flatten(hum)
         if n==6:
             p.append(final_array[6])
             f_p= flatten(p)
         if n==7:
             l.append(final_array[7])
             f_l= flatten(l)  
         if n==8:
           data_index.append(final_array[8])
           f_data_index= flatten(data_index)      
                                                        
      lists= zip(f_date,f_time,f_sm1,f_sm2,f_temp,f_hum,f_l,f_p,f_data_index)              
      for row in lists:
          writer.writerow(row) 
             #try_array=[]
       #for n in range(9):
##   print(final_array[n],n)
                            