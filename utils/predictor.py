import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

scaler=MinMaxScaler(feature_range=(0,1))

def train_model(df):
    data=df[['Close']].values
    scaled=scaler.fit_transform(data)

    x=[]
    y=[]

    for i in range(60,len(scaled)):
        x.append(scaled[i-60:i,0])
        y.append(scaled[i,0])

    x=np.array(x)
    y=np.array(y)

    x=np.reshape(x,(x.shape[0],x.shape[1],1))

    model=Sequential()
    model.add(LSTM(64,return_sequences=True,input_shape=(x.shape[1],1)))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x,y,batch_size=16,epochs=3,verbose=0)

    return model,scaled