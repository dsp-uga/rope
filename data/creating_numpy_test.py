import glob
import os
import numpy as np 
import csv
import cv2
a=glob.glob('/home/rdey/dsp_final/test/*.jpg')
X_test=[]

#print(a[0].replace('/home/rdey/dsp_final/train/','').replace('.jpg',''))
for i in range (0,len(a)):
    if(i%10000==0):
        print("current=",i)
    try:

        #print(('/home/rdey/dsp_final/train/'+str(a[i].replace('/home/rdey/dsp_final/train/','').strip())))
        temp_x=cv2.imread(('/home/rdey/dsp_final/test/'+str(a[i].replace('/home/rdey/dsp_final/test/','').strip())),1)
        
        X_test.append(cv2.resize(temp_x,(64,64)))
    
                
    except:
        print('error',i)

np.save('X_test.npy',X_test)

