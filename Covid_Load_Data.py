import urllib.request
import numpy as np
import os
import zipfile
import pandas
import shutil
import requests
crawling='http://fifaon3.cafe24.com/562468_1022626_bundle_archive.zip'#
crawling1='http://fifaon3.cafe24.com/738672_1280224_bundle_archive.zip'#코로나만
savename='covid.zip'
savename1='covidd.zip'

firstfds=urllib.request.urlretrieve(crawling)
with zipfile.ZipFile("covid.zip") as za:
    za.extractall()

sefsda=urllib.request.urlretrieve(crawling1)
with zipfile.ZipFile("covidd.zip") as zf:
    zf.extractall()

now_path=os.getcwd()
loacl_path=[]
for i in range(len(now_path)):
    loacl_path.append(now_path[i])
for j in range(len(loacl_path)):
    if loacl_path[j]=='\\':
        loacl_path[j]='/'
now_path="".join(loacl_path)# 기본경로
first=now_path+'/562468_1022626_bundle_archive/'
second=now_path+'/738672_1280224_bundle_archive/'

chest_data=pandas.read_csv(first+'Chest_xray_Corona_Metadata.csv',header=None)
chest_data1=pandas.read_csv(second+'/metadata.csv',header=None)
chest_data=np.array(chest_data)
chest_data1=np.array(chest_data1)

path = os.path.join(now_path, 'covid')
path1 = os.path.join(path, 'train')
path2 = os.path.join(path, 'test')
path3=os.path.join(path1,'Normal')
path4=os.path.join(path2,'Normal')
path5=os.path.join(path1,'Virus')
path6=os.path.join(path2,'Virus')
path7=os.path.join(path1,'bacteria')
path8=os.path.join(path2,'bacteria')
path9=os.path.join(path1,'COVID_19')
path10=os.path.join(path2,'COVID_19')
os.mkdir(path)
os.mkdir(path1)
os.mkdir(path2)
os.mkdir(path3)
os.mkdir(path4)
os.mkdir(path5)
os.mkdir(path6)
os.mkdir(path7)
os.mkdir(path8)
os.mkdir(path9)
os.mkdir(path10)

image_train_path=first+'/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
image_test_path=first+'/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'
image_train_path1=second+'images/'

for i in range(1,len(chest_data)):
    try :
        if chest_data[i,3]=='TRAIN':
            if chest_data[i,2]=='Normal':
                shutil.move(image_train_path+'{}'.format(chest_data[i,1]),path3)
            elif chest_data[i,5]=='bacteria':
                shutil.move(image_train_path+'{}'.format(chest_data[i,1]),path7)
            elif chest_data[i,5]=='Virus':
                shutil.move(image_train_path+'{}'.format(chest_data[i,1]),path5)
            elif chest_data[i,5]=='COVID-19':
                shutil.move(image_train_path+'{}'.format(chest_data[i,1]),path9)
        elif chest_data[i,3]=='TEST':
            if chest_data[i,2]=='Normal':
                shutil.move(image_test_path+'{}'.format(chest_data[i,1]),path4)
            elif chest_data[i,5]=='bacteria':
                shutil.move(image_test_path+'{}'.format(chest_data[i,1]),path8)
            elif chest_data[i,5]=='Virus':
                shutil.move(image_test_path+'{}'.format(chest_data[i,1]),path6)
            elif chest_data[i,5]=='COVID-19':
                shutil.move(image_test_path+'{}'.format(chest_data[i,1]),path10)
    except :
        print('pass')
        pass
num=0
for m in range(1,len(chest_data1)):
    num+=1
    try:
        if num<250:
            if chest_data1[m,4]=='COVID-19':
                shutil.move(image_train_path+'{}'.format(chest_data1[m,22]),path10)
        else:
            if chest_data1[m, 4] == 'COVID-19':
                shutil.move(image_train_path1 + '{}'.format(chest_data1[m, 22]), path10)
    except :
        print('pass')
