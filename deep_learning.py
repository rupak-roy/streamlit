#semi auto ML app
import streamlit as st


#data visualization pcks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#import plotly.graph_objects as go
import plotly.express as px


#LSTM for regression 
import numpy
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#disable warning message
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)



def run_deep_learning():
    st.subheader("Welcome to Deep Learning Analytics")

    data = st.file_uploader("Upload Time-Series Data",type = ["csv","txt"])
    if data is not None:
        dp = pd.read_csv(data,sep=',')
        st.dataframe(dp.head())
            
        if st.sidebar.checkbox("Show Dimensions(Shape)"):
            st.success("In (Row, Column) format")
            st.write(dp.shape)
                
        if st.sidebar.checkbox("Data Types"):
            st.success("In (Column Names , Data Type) format")
            st.table(dp.dtypes)
            
        if st.sidebar.checkbox("Show Missing Values"):
            st.write(dp.isnull().sum())
        
        if st.sidebar.checkbox("Auto-Impute Missing Values*"):
            st.info("Currently support ~ dropna | droping the missing values")
            dp = dp.dropna()
            st.write(dp.isnull().sum())
            
        if st.sidebar.checkbox("Remove Columns*"):
            all_columns = dp.columns.to_list()
            selected_columns = st.sidebar.multiselect("Remove columns like 'Date'. Keep only one column you wish to predict",all_columns)
            #st.write(str(selected_columns))
            global new_drop_dp
            new_drop_dp = dp.drop(selected_columns,axis=1)
            st.write(new_drop_dp)
            st.success("Auto Data Pre-processing is complete, let's build the deep learning model ")
            st.warning("Note: Deep Learning Model take time to get train, so please be patience.")

        
    if st.sidebar.checkbox("ANN Analytics"):
        st.write("coming soon")
    
    if st.sidebar.checkbox("LSTM Analytics"):

        lstm_algo = ['LSTM with regression','LSTM with Window Method','LSTM with TimeStep Framing','LSTM with Memory Between Batches','Stacked LSTM with Memory Between Batches']
        classifier = st.selectbox("Select the type of LSTM Network",lstm_algo)
        
        if  classifier == 'LSTM with regression':
            numpy.random.seed(7)
            # load the dataset
            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            # reshape into X=t and Y=t+1
            look_back = 1
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)

            # reshape input to be [samples, time steps, features]
            trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            # create and fit our data to the LSTM network
            model = Sequential()
            model.add(LSTM(30, input_shape=(1, look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='Adagrad',metrics =["accuracy"])
            model.fit(trainX, trainY, epochs=370, batch_size=28, verbose=1)

            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            
            l01,l02 = st.beta_columns(2)
            with l01:
                st.success("The Predicted Results")
                st.write(testPredict)
            with l02:
                st.info("The Loss Function Score")
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score in RMSE',trainScore)
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score in RMSE',testScore)
            
            #-----Visualize---------- 
            st.info("Let's Visualize the results ")
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.legend(['The dataset','Train dataset','Test dataset'])
            st.pyplot()
            
            st.info("")
            st.success("")

#-----------------LSTM with Window Method-----------------------------------        
        if  classifier == 'LSTM with Window Method':
            numpy.random.seed(8)
            # load the dataset
            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            st.info("")
            st.success("")
            st.write("LSTM with window method starts here")
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            # reshape into X=t and Y=t+1
            look_back = 3 #window-method (t-2,t-1,t,y)

            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            # reshape input to be [samples, time steps, features]
            trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
            
            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(17, input_shape=(1, look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
            model.fit(trainX, trainY, epochs=450, batch_size=28, verbose=1)

            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
        
            l03,l04 = st.beta_columns(2)
            with l03:
                st.success("The Predicted Results")
                st.write(testPredict)
            with l04:
                st.info("The Loss Function Score")
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score in RMSE',trainScore)
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score in RMSE',testScore)
               
        #-----------------Visualize----------------
            st.info("Let's Visualize the results ")
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.legend(['The dataset','Train dataset','Test dataset'])
            st.pyplot()
        
        if  classifier == 'LSTM with TimeStep Framing':
            numpy.random.seed(8)
            # load the dataset
            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            st.info("")
            st.success("")

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            # reshape into X=t and Y=t+1
            look_back = 3
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)

            # reshape input to be [samples, time steps, features]
            trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
            #we are putting back the feature dimension = 1
            
            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(17, input_shape=(look_back, 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainY, epochs=450, batch_size=28, verbose=0)

           # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
               
            l04,l06 = st.beta_columns(2)
            with l04:
                st.success("The Predicted Results")
                st.write(testPredict)
            with l06:
                st.info("The Loss Function Score")
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score in RMSE',trainScore)
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score in RMSE',testScore)
               
            #----------Visualize-------------------------
            st.info("Let's Visualize the results ")
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
               
            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.legend(['The dataset','Train dataset','Test dataset'])
            st.pyplot()

        if  classifier == 'LSTM with Memory Between Batches':
            numpy.random.seed(8)
            # load the dataset
            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            st.info("")
            st.success("")

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]               
            
            #reshape into X=t and Y=t+1
            look_back = 3
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)

            #reshape input to be [samples, time steps, features]
            trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

            #the LSTM network
            batch_size = 1
            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            for i in range(100):
                model.fit(trainX, trainY, epochs=10, batch_size=batch_size, verbose=2, shuffle=False)
                model.reset_states()
      
    
            # make predictions
            trainPredict = model.predict(trainX, batch_size=batch_size)
            model.reset_states()
            testPredict = model.predict(testX, batch_size=batch_size)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])

            l07,l08 = st.beta_columns(2)
            with l07:
                st.success("The Predicted Results")
                st.write(testPredict)
            with l08:
                st.info("The Loss Function Score")
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score in RMSE',trainScore)
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score in RMSE',testScore)

            #----------Visualize-----------------------------------------------------
            st.info("Let's Visualize the results ")
            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.legend(['The dataset','Train dataset','Test dataset'])
            st.pyplot()           

        if  classifier == 'Stacked LSTM with Memory Between Batches':
            numpy.random.seed(8)
            # load the dataset
            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            st.info("")
            st.success("")

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] 
            
            #reshape into X=t and Y=t+1
            look_back = 3
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            # reshape input to be [samples, time steps, features]
            trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

            # create and fit the LSTM network
            batch_size = 1
            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            for i in range(100):
                model.fit(trainX, trainY, epochs=20, batch_size=batch_size, verbose=2, shuffle=False)
                model.reset_states()
            #make predictions
            trainPredict = model.predict(trainX, batch_size=batch_size)
            model.reset_states()
            testPredict = model.predict(testX, batch_size=batch_size)
            #invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            
            l09,l10 = st.beta_columns(2)
            with l09:
                st.success("The Predicted Results")
                st.write(testPredict)
            with l10:
                st.info("The Loss Function Score")
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score in RMSE',trainScore)
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score in RMSE',testScore)
                
            st.info("Let's Visualize the results ")
            #shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            #shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

            #plot baseline and predictions
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.legend(['The dataset','Train dataset','Test dataset'])
            st.pyplot()      

        
        
    if st.sidebar.checkbox("CNN Analytics"):
        st.write("coming soon")
        
        
        

        
        