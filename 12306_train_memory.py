from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,BatchNormalization,Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import  EarlyStopping

import numpy as np

ROW = 66
COL = 66


'''
setInterval(()=>{
	if(Array.from(document.getElementById("connect").children[0].children[2].innerHTML).splice(3,4).toString() === '重,新,连,接'){
		document.getElementById("connect").children[0].children[2].click()
	}
},1000)
'''

#colab  https://colab.research.google.com/drive/1cMvfCbfzMl7zLiNyTPPOyIvWmBkeBjBQ
'''
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive
import os
import sys
os.chdir('drive/Colab Notebooks')
'''

import os
def get_google_drive_path():
    os.chdir(os.path.join(os.getcwd(), 'drive'))
    print(os.getcwd())
    print(os.listdir(os.getcwd()))

##准备数据
def normalize(image):
    out = np.array(image,dtype='float32')
    out /= 255.0
    return out

def load_data():
    with np.load("data_simple.npz", allow_pickle=True) as f:
        x, y = f['x'], f['y']
    return x,y

x_train,y_train = load_data()
x_train /= 255
y_train = np_utils.to_categorical(y_train,num_classes=80)

(trainX, testX, trainY, testY) = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.99,
                                                  random_state=42)

image_generaotr = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")


model = Sequential([
    Conv2D(input_shape=(ROW, COL, 3), filters=32, kernel_size=(3, 3), padding='same', activation='relu',
           data_format='channels_last'),
    # Activation('relu'),
    BatchNormalization(),
    MaxPool2D(padding="same", pool_size=(3, 3)),
    Dropout(0.1),

    Conv2D(input_shape=(ROW, COL, 3), filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPool2D(padding="same", pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(input_shape=(ROW, COL, 3), filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    MaxPool2D(padding="same", pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(input_shape=(ROW, COL, 3), filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    MaxPool2D(padding="same", pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(input_shape=(ROW, COL, 3), filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
    # Activation('relu'),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    # Activation('relu'),
    BatchNormalization(),
    MaxPool2D(padding="same"),
    Dropout(0.1),

    Flatten(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(80, activation='softmax'),
]
)
'''
from keras.layers import Activation,MaxPooling2D
inputShape = (ROW, COL, 3)
chanDim=-1
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(80))
model.add(Activation("softmax"))

'''

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=64,epochs=1,validation_split=0.2)# 0.2percent for validate

# history = model.fit_generator(image_generaotr.flow(trainX, trainY, batch_size=1),validation_data=(testX, testY),
#     steps_per_epoch=len(x_train),
#     epochs=1,
#     verbose=1,
#     callbacks=[EarlyStopping(patience=50)])



import matplotlib.pyplot as plt

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Model Accuracy")
key = 'accuracy'
val_key = 'val_accuracy'
if 'acc' in history.history.keys():
    key = 'acc'
    val_key = 'val_acc'

plt.plot(history.history[key])
plt.plot(history.history[val_key])

plt.legend(['train,test'],loc='best')

plt.show()


model.save("12306.h5")
