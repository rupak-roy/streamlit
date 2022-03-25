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


#-------------LSTM FORECASTING -----------

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

from numpy import concatenate
from sklearn.preprocessing import LabelEncoder

#----------END


#-----CNN pckgs
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import LeakyReLU
from keras.layers import TimeDistributed
import numpy as np
import pandas as pd


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

#----------- LSTM FORECASTING DEF----------------
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
#--------------------------LSTN FORECASTING DEF ENDS-----------------

#------- LSTM Multi-Step--------------
# convert series to supervised learning
def series_to_supervised_multi_step(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#------------------ Multi-Step -------Ends


#------------------MULTI-STEP LSTM FORECASTING-----------------------------------------------
#a.) convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
#b.) create a differenced series
def difference_multi(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
#c.) transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test
 
#d.) Define the LSTM network 
def fit_lstm_multi(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		model.reset_states()
	return model
 
#e.) forecast with an LSTM,
def forecast_lstm_multi(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
#f.)  evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm_multi(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
#g.) invert differenced forecast
def inverse_difference_multi(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted
 
#h.) inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference_multi(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted
 
#i.) evaluate the model with RMSE 
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		st.write("RMSE SCORE",(rmse))
        #print('t+%d RMSE: %f' % ((i+1), rmse))
 
#j.) plot the forecasts
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	    #st.pyplot()
    # show the plot
	pyplot.show()
    #st.pyplot()
    #st.pyplot()
#-----------------------------------------------------------
 

def run_deep_learning():
    st.subheader("Welcome to Deep Learning Analytics - beta phase")

    data = st.file_uploader("Upload Time-Series Data",type = ["csv","txt"])
    if data is not None:
        dp = pd.read_csv(data,sep=",")
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
            
        if st.sidebar.checkbox("Select The Column*"):
            all_columns = dp.columns.to_list()
            selected_columns = st.sidebar.multiselect("Select the Time-Series column like 'sales'. Keep only one column you wish to predict",all_columns)
            #st.write(str(selected_columns))
            global new_drop_dp
            new_drop_dp = dp[selected_columns]
            st.write(new_drop_dp)
            st.success("Auto Data Pre-processing is complete, let's build the deep learning model ")
            st.warning("Note: Deep Learning Model take time to get train, so please be patience.")

        
    if st.sidebar.checkbox("ANN Analytics"):
        st.write("coming soon")
    
    if st.sidebar.checkbox("LSTM Analytics"):

        lstm_algo = ['LSTM with regression','LSTM with Window Method','LSTM with TimeStep Framing','LSTM with Memory Between Batches','Stacked LSTM with Memory Between Batches'
                     ,'LSTM Forecasting-Univariate','LSTM Forecasting-Multi-Step','LSTM Forecasting-Multivariate']
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
        
        if  classifier =='LSTM Forecasting-Univariate':

            dataframe = new_drop_dp
            dataset = dataframe.values
            dataset = dataset.astype('int64')
            
            #series = pd.DataFrame(dataframe)
            # load dataset
            #series = dataset

            # transform data to be stationary
            raw_values = dataset            
            diff_values = difference(raw_values, 1)
 
            # transform data to be supervised learning
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
             
            # split data into train and test-sets
            train, test = supervised_values[0:-12], supervised_values[-12:]
    
            # transform the scale of the data
            scaler, train_scaled, test_scaled = scale(train, test)
 
            # fit the model
            lstm_model = fit_lstm(train_scaled, 1,500, 14)
            # forecast the entire training dataset to build up state for forecasting
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=1)
            
            # walk-forward validation on the test data
            predictions = list()
            for i in range(len(test_scaled)):
                # make one-step forecast
                X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
                yhat = forecast_lstm(lstm_model, 1, X)
                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # invert differencing
                yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
                # store forecast
                predictions.append(yhat)
                expected = raw_values[len(train) + i + 1]
                st.write(" Month,Predicted,Expected",(i+1, yhat, expected))
                #st.write( print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected)))
                #st.write(p)
                
            # report performance #raw_values[-12,] refers last 12 months/rows 
            rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
            st.write('Test RMSE Score' , rmse)
            # line plot of observed vs predicted
            pyplot.plot(raw_values[-12:])
            pyplot.plot(predictions)
            pyplot.legend(['Actual','Prediction'])
            st.pyplot()
        
        if  classifier =="LSTM Forecasting-Multi-Step":
            
            # configure
            n_lag = 1
            n_seq = 3
            n_test = 10
            n_epochs = 1500
            n_batch = 1
            n_neurons = 50
            
            
            series = new_drop_dp
            #dataset = dataframe.values
            #dataset = dataset.astype('float64')

            #series = read_csv('sales_year.csv', usecols=[1], engine='python')
            #prepare data
            scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
            #fit model
            model = fit_lstm_multi(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
            
            #forecasts
            forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
            
            #inverse transform forecasts and test
            forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
            actual = [row[n_lag:] for row in test]
            actual = inverse_transform(series, actual, scaler, n_test+2)
            
            #evaluate forecasts
            evaluate_forecasts(actual, forecasts, n_lag, n_seq)
            
            #plot forecasts
            plot_forecasts(series, forecasts, n_test+2)
        
        if  classifier =="LSTM Forecasting-Multivariate":
            st.write("Please select multiple variables.As of now select all columns except Date from the sample dataset for this Demo version")
            # load dataset
            dataset = new_drop_dp
            #dataset = read_csv('pollution.csv', header=0, index_col=0)
            values = dataset.values
            # integer encode direction
            encoder = LabelEncoder()
            values[:,4] = encoder.fit_transform(values[:,4])
            # ensure all data is float
            values = values.astype('float32')
            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1)) #(dew and temp column is in negative)
            scaled = scaler.fit_transform(values)
            # frame as supervised learning
            reframed = series_to_supervised(scaled, 1, 1)
            # drop columns we don't want to predict
            reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
            
            st.write(reframed.head())
            
            # split into train and test sets
            values = reframed.values
            n_train_hours = 365 * 24 #we will use the value as index/row number
            train = values[:n_train_hours, :]
            test = values[n_train_hours:, :]
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features] 8features i.e. 8 columns
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
     
            # design network
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            # fit network
            history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=1, shuffle=False)
            
            # plot history
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            #pyplot.show()
            st.pyplot()
            
            # make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            # invert scaling for forecast
            inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            # invert scaling for actual
            test_y = test_y.reshape((len(test_y), 1))
            inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]
            # calculate RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            st.write('Test RMSE:', rmse)             

        
        
    if st.sidebar.checkbox("CNN Analytics"):
        # split a univariate sequence into samples
        def split_sequence(sequence, n_steps_in, n_steps_out):
            X, y = list(), list()
            for i in range(len(sequence)):
                # the end of this pattern
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out
		        # check if we are beyond the sequence
                if out_end_ix > len(sequence):
                    break
                #gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)
               
        # load the dataset
        #st.write(dp.head())
        st.info("Let's format our data for CNN")
        
        if st.checkbox("Auto-Impute Missing Values*",key='cnn101'):
            st.success("Currently support ~ dropna | droping the missing values")
            dp = dp.dropna()
            st.write(dp.isnull().sum())
            
        if st.checkbox("Select the Column*",key='cnn102'):
            all_columns = dp.columns.to_list()
            selected_columns_cnn = st.multiselect("Select the column(time series). Keep only one column you wish to predict",all_columns)
            #st.write(str(selected_columns))
            global new_drop_dp_cnn
            new_drop_dp_cnn = dp[selected_columns_cnn]
            st.write(new_drop_dp_cnn.head())
            st.success("Auto Data Pre-processing is complete, let's build the deep learning model ")
            st.warning("Note: Deep Learning Model take time to get train, so please be patience.")
            
        if st.checkbox("Analyse using Convolutional Neural Network"):

            raw_seq = new_drop_dp_cnn.values
            #cant write too long data it hangs st.write(raw_seq)        
            #after converting to list ,it shows the original value as str
            raw_seq = raw_seq.tolist()

            # choose a number of time steps
            n_steps_in, n_steps_out = 3,24            
            # split into samples
            X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
            
            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            
            #---------testig till here
            
            
            # define model
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(n_steps_out))
            model.compile(optimizer='adam', loss='mse',metrics = ["accuracy"])
            # fit model
            #model.fit(X, y, epochs=50, verbose=0)
            st.error("Streamlit Error")
        


        
        