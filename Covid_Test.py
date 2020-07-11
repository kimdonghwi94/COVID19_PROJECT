import numpy as np
import random
from keras.preprocessing import image
import os
import keras
img_width, img_height = 299, 299
now_path=os.getcwd()
loacl_path=[]
for i in range(len(now_path)):
    loacl_path.append(now_path[i])
for j in range(len(loacl_path)):
    if loacl_path[j]=='\\':
        loacl_path[j]='/'
now_path="".join(loacl_path)

model = keras.models.load_model(now_path+'/donghwi_COVID19')

train_dir=now_path+'/covid/train'
test_dir=now_path+'/covid/test'
bac=test_dir+'/bacteria/'
cov=test_dir+'/COVID_19/'
nor=test_dir+'/Normal/'
Virus=test_dir+'/Virus/'
a=[]
b=[]
c=[]
d=[]
conv_base = keras.applications.InceptionV3(weights='imagenet',include_top = False,input_shape=(img_width, img_height, 3))
def visualize_predictions(classifier, n_cases):
    for i in range(0,n_cases):
        path = random.choice([bac,cov,nor,Virus]) #테스트데이터 경로 랜덤으로 선택
        random_img = random.choice(os.listdir(path))#
        img_path = os.path.join(path, random_img)# 테스트 이미지 랜덤으로 선택
        img = image.load_img(img_path, target_size=(img_width, img_height)) #랜덤으로 가져온 데이터 전처리
        img_tensor = image.img_to_array(img)
        img_tensor /= 255. #데이터 0~1사이값으로 전처리
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))
        prediction = classifier.predict(features)
        prediction=np.argmax(prediction)
        print(img_path[46],img_path,prediction)
        #0: 'COVID_19', 1: 'Normal', 2: 'Virus', 3: 'bacteria'
        if  img_path[46]=='b' and prediction==3 :   #a=박테리아
            a.append(1)                             #b=코로나
        elif  img_path[46]=='C' and prediction==0 : #c=노말
            b.append(1)                             #d=바이러스
        elif  img_path[46]=='N' and prediction==1:
            c.append(1)
        elif  img_path[46]=='V' and prediction==2:
            d.append(1)
        elif  img_path[46]=='b' and prediction!=3 :
            a.append(0)
        elif  img_path[46]=='C' and prediction!=0:
            b.append(0)
        elif  img_path[46]=='N' and prediction!=1 :
            c.append(0)
        elif  img_path[46]=='V' and prediction!=2:
            d.append(0)
    print(a,b,c,d)
    print('\n')
    print(np.mean(a),np.mean(b),np.mean(c),np.mean(d))
    print('\n')
    print((np.mean(a)+np.mean(b)+np.mean(c)+np.mean(d))/4)
    return a,b,c,d

e,f,g,h=visualize_predictions(model, 890)  # 테스트할 이미지들의 갯수 설정가능
e=np.array(e)
f=np.array(f)
g=np.array(g)
h=np.array(h)
save1=np.save('./bacteria',e)
save2=np.save('./covid',f)
save3=np.save('./normal',g)
save4=np.save('./virus',h)