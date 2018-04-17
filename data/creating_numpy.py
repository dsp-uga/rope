import glob
import os
import numpy as np 
import csv
import cv2

train_path = 'train/'
output_path = 'train_np/'

a=glob.glob( os.path.join( train_path, '*.jpg'))
X_train=[]
y_train=[]
counter=0
csv_file=csv.reader(open('/home/rdey/dsp_final/train.csv'),delimiter=',')
csv_file_1=dict((rows[0],rows[2])for rows in csv_file)
csv_file=csv_file_1
#print(csv_file[0])
#print(a[0].replace('/home/rdey/dsp_final/train/','').replace('.jpg',''))



counter=0
to_skip=0
for i in range (0,len(a)):
    
    if(i%100000==0 and to_skip==1):
        print("current=",i)
    try:

        #print(('/home/rdey/dsp_final/train/'+str(a[i].replace('/home/rdey/dsp_final/train/','').strip())))
        temp_x=cv2.imread((train_path+str(a[i].replace(train_path,'').strip())),1)
        

        try:

                
            y_train.append(csv_file[str(a[i].replace(train_path,'').replace('.jpg',''))])
            X_train.append(cv2.resize(temp_x,(64,64)))
        except:

            print("cant find entry")
        
        if(i%10000==0 and to_skip==1):
            np.save(  output_path+ '/X_train'+str(counter)+'.npy',np.array(X_train))
            np.save(output_path+'/y_train'+str(counter)+'.npy',np.array(y_train))
            counter+=1
            X_train=[]
            y_train=[]
        
    


    except:
        print('error',i)
    to_skip=1
    

np.save('X_train_last.npy',X_train)
np.save('y_train_last.npy',y_train)

