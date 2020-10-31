#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


# In[6]:


df = pd.read_csv("Indonesian Sentiment Twitter Dataset Labeled.csv")
df.head(10)


# In[7]:


text= df["tweet"].tolist()

print(len(text))


# In[8]:


y = df["sentimen"]
y = to_categorical(y)
print(y)


# In[9]:


df["sentimen"].value_counts()


# In[10]:


token = Tokenizer()
token.fit_on_texts(text)
token.index_word


# In[11]:


vocab = len(token.index_word)+1
vocab


# In[12]:


encode_text = token.texts_to_sequences(text)


# In[13]:


max_kata = 100
X = pad_sequences(encode_text, maxlen=max_kata, padding="post")
X


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X,y, random_state = 40, test_size=0.3, stratify =y)


# In[15]:


X_train = np.asarray(X_train) 
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

print(len(X_train))
print(len(X_test))
print(len(Y_test))
print(len(Y_train))


# In[16]:


voc_size = 300
model = Sequential()
model.add(Embedding(vocab, voc_size,input_length=max_kata))
model.add(Conv1D(64,8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16, activation="relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(3,activation="softmax"))
model.summary()


# In[17]:


model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss= "categorical_crossentropy", metrics=["accuracy"])


# In[18]:


model.fit(X_train,Y_train, epochs=3, validation_data=(X_test,Y_test))


# In[ ]:


def get_encode(x):
  x = token.texts_to_sequences(x)
  x = pad_sequences(x, maxlen = max_kata, padding="post")
  return(x)


# In[ ]:


coba = ["Setelah dikabarkan bahwa dirinya dikeluarkan dari Partai Amanat Nasional Amien Rais kini mendirikan partai baru Dimana setelah melalui diskusi yang panjang dari 28 nama sudah ditentukan sebuah nama untuk partai barunya tersebut yaitu PAN Reformasi Selain memiliki nama yang mirip dengan nama partai sebelumnya partai baru Amien Rais yang direncanakan akan diperkenalkan pada Desember 2020 tersebut pun juga memiliki logo yang tidak jauh berbeda dengan PAN Yakni berwarna biru serta terdapat matahari putih Mengenai siapa yang akan menduduki kursi kepemimpinan partai barunya tersebut Amien Rais mengakui bahwa dirinya akan menyerahkan area tersebut kepada generas generasi yang lebih muda Diantaranya ChandraTurta Wijaya juga Putra Jasa Husein Seperti diketahui sebelumnya alasan politikus senior ini tidak lagi bergabung bersama PAN dan mendirikan partai baru dikarenakan prinsipnya sudah tidak sejalan lagi dengan PAN yang saat ini dipimpin oleh Zulfikli Hasan Dimana PAN saat ini menyatakan akan memberi dukungan penuh terhadap pemerintahan Joko Widodo sedangkan Amien Rais merupakan salah satu tokoh yang anti rezim Jokowi Bukan Amien Rais namanya kalau tidak punya bahan untuk megkritik secara pedas pemerintahan Jokowi Baru baru ini pada akun sosial media instagram miliknya Amien kerap mengunggah berbagai kritik kepada pemerintah yang telah dikemasnya sedemikian rupa untuk ditampilkan kepada khalayak Dalam hal ini Amien memberikan tanggapan terkait perkembangan politik nasional di era Presiden Jokowi Menurutnya politik saat ini hanya memecah belah persatuan dan kesatuan bangsa Indonesia Namun jika kita cermati lebih jauh mengenai kritikan tersebut apakah benar politik era Jokowi kurang demokratis namun di sisi lain Amien Rais tetap bebas menyuarakan kritikannya terhadap Jokowi Presiden Jokowi memang telah menunjukan bahwa dirinya bukanlah sosok yang anti kritik Namun dalam berdemokrasi kritik haruslah berdasarkan oleh data yang lengkap dan diserai solusi terhadap masalah yang tengah dikritik Thanks for all your Support For Business Politindo com Facebook Instagram"]
x= get_encode(coba)
model.predict_classes(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




