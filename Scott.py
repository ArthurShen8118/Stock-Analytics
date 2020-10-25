#1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas
df = pandas.read_csv('https://raw.githubusercontent.com/ScottHsieh-SH/1st-DL-CVMarathon/master/999.csv', encoding = 'utf-8' )


#2

X= df.iloc[23:399, 1:-2].values
y = df.iloc[23:399, -1].values





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





#3

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#4


#Part2

#Import
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the ANN

classifier = Sequential()


#Input Layer and Hidden layer 


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dropout(p = 0.1))
#Second Hidden Layer 

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))



#Output Layer 

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#Compling the ANN (how to optimize weight)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])





#Fitting the ANN to the Trainning set 



train_history=classifier.fit(X_train, y_train, validation_split=0.2, batch_size = 10, epochs = 100)




#5


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix-
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 


accuracy(cm)


###########################PLot###############################################





def show_train_history (train_history,train,validation):
    
    
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

show_train_history(train_history,'loss','val_loss')

img.save('C:\Users\arthur.shen\Desktopmyphoto.jpg', 'JPEG')

###########################CM PLot###############################################




from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np




df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (2,2))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})























