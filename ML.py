#semi auto ML app
import streamlit as st


#EDA pkgs
import pandas as pd
import numpy as np

#data visualization pcks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#import plotly.graph_objects as go
import plotly.express as px


#ML pckgs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
#disable warning message
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

import pickle

import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
# Fxn to Download Result
def download_link(object_to_download, download_filename, download_link_text):
    d=pickle.dump(lr_model, open(filename, 'wb'))
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    return f'<a href="data:file/txt;base64,{b64}" download="{d}">"click click"</a>'

def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "Analysis_report_{}_.csv".format(timestr)
    st.markdown("🤘🏻  Download CSV file ⬇️  ")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    """Auto-Machine Learning App with Streamlit """
    
    st.title("Auto-Machine Learning Data Analytics Kit .v1 - full version")
    st.text( "10 Powerful classifiers with STreamlit, upload data and enjoy ")
    st.write("""
             Sample Dataset [@rupak-roy Github](https://github.com/rupak-roy/dataset-streamlit) . V2 update: GridSearch 
             """)
    st.text("~ Currently accepting smaller files due to Heroku free storage limit, will be transfered to streamlit platform soon ")
  
    st.text("Ignore:EmptyDataError: No columns to parse from file have no effect on analysis,This is indentation Error will be fixed in next update")
   
    activities = ["EDA","Plots","Model Building","About"]
    
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            st.dataframe(df.head(20))
            
            if st.sidebar.checkbox("Show Dimensions(Shape)"):
                st.success("In (Row, Column) format")
                st.write(df.shape)
                
            if st.sidebar.checkbox("Data Types"):
                st.success("In (Column Names , Data Type) format")
                st.table(df.dtypes)
            
            if st.sidebar.checkbox("Show Columns Names"):
                st.success("Column Names")
                all_columns = df.columns.to_list()
                st.write(all_columns)
                
            if st.sidebar.checkbox("Summary Statistics"):
                st.table(df.describe())
                
            if st.sidebar.checkbox("Show Missing Values"):
                st.write(df.isnull().sum())   
                              
                
            if st.sidebar.checkbox("Remove Columns*"):
                all_columns = df.columns.to_list()
                selected_columns = st.sidebar.multiselect("Select Columns like ID to remove",all_columns)
                #st.write(str(selected_columns))
                new_drop_df = df.drop(selected_columns,axis=1)
                st.write(new_drop_df)   
           
            
            if st.sidebar.checkbox("Select Columns to Show*"):
                all_columns = new_drop_df.columns.to_list()
                selected_columns = st.sidebar.multiselect("Select Columns",all_columns)
                new_df = new_drop_df[selected_columns ]
                st.dataframe(new_df)

            if st.sidebar.checkbox("Value Counts on Selected Columns"):
               st.write(new_df.value_counts())
                             
            if st.sidebar.checkbox("Pivot Table by Average"):
                all_columns_names = df.columns.tolist()
                sca1, sca2 = st.beta_columns(2)
                with sca1:
                    X = st.multiselect("Select Column for Values",all_columns_names,key='0033')
                with sca2:
                    Y = st.multiselect("Select Column for Index",all_columns_names,key='0044')
                               
                impute_grps = df.pivot_table(values=X, index=Y, aggfunc=np.mean)
                st.dataframe(impute_grps)

            #if st.sidebar.checkbox('Cross-Tab Analysis'):
              #  all_columns_names = df.columns.tolist()
               # cross1, cross2 = st.beta_columns(2)
                #with cross1:
                 #   X = st.multiselect("Select Column for Cross Tab",all_columns_names,key='0033')
                #with cross2:
                 #   Y = st.multiselect("Select Column for Cross Tab",all_columns_names,key='0044')              
                #cc=pd.Series(X) 
                #X1 = df[X]
                #need to convert to list then series
                #c1=X1.values.tolist()
                #c2 = pd.Series(c1)
                #Y1 = df[Y]
                #c_data = pd.crosstab(c2,Y1.Loan_Status,margins=True)
                #st.write(c_data)
                
            if st.sidebar.checkbox("Corelation with Seaborn using selected columns"):
                st.write(sns.heatmap(new_df.corr(),annot=True))
                st.pyplot()    
        
            if st.sidebar.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
            #need to use a condition else it will render 1st column before selection n takes him time
                columns_to_plot= st.sidebar.selectbox("Select the column*", all_columns)
                if st.sidebar.checkbox("Render Pie Chart"):
                    pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie_plot) 
                    st.pyplot()
        
        
    elif choice == 'Plots':
        st.subheader("Data Visualization")
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            st.dataframe(df.head())
        else:
            st.error("Error: upload your data")
                                      
        all_columns_names = df.columns.tolist()
        type_of_plot = st.sidebar.selectbox("Select type fo plot",["Distribution Plot","Area","Bar","Line","Hist"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
        
        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
        
        #plot by Streamlit
        if type_of_plot == 'Area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)
            
        elif type_of_plot =='Bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)
        
        elif type_of_plot =='Line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
            
        elif type_of_plot =='Distribution Plot':
            cust_data = df[selected_columns_names]
            sns.distplot(cust_data);
            st.pyplot()
            
        elif type_of_plot =='Hist':
            cust_data = df[selected_columns_names]
            plt.hist(cust_data)
            st.pyplot()
        
           
        elif type_of_plot =='Box':
            cust_data = df[selected_columns_names]
            plt.box(cust_data)
            st.pyplot()
            
        elif type_of_plot =='kde':
            cust_data = df[selected_columns_names]
            plt.kde(cust_data)
            st.pyplot()
            
        if st.sidebar.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            columns_to_plot= st.sidebar.selectbox("Select the column*", all_columns)
            if st.sidebar.checkbox("Render Pie Chart"):
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot) 
                st.pyplot()
            
            
#------ SCATTER PLOT--------------------------        
        if st.sidebar.checkbox("Scatter Plot"):
            st.write("Scatter Plot")
            all_columns_names = df.columns.tolist()
        
        #variety= st.multiselect("Select 1 Category Column To Plot (optinal)",all_columns_names,key='002')
        #cat_col = df[variety]
        #col_nam = str(cat_col.columns)
        #n=cat_col['Species'].unique()
        #n=pd.Series(n)
        #categoies = st.multiselect('Show iris per variety?',n)
        #X = st.multiselect("Select 1 Column X To Plot",all_columns_names,key='0033')
        #Y = st.multiselect("Select 1 Column y To Plot",all_columns_names,key='0044')
        #X1 = df[X]
        #Y1 = df[Y]
        #new_df = df[(df[n].isin(n))]
        
        #fig = px.scatter(df,x =X1,y=Y1,color='n')
        #st.plotly_chart(fig)
            #Scatter Plot Layouts
            sca1, sca2 = st.beta_columns(2)
            with sca1:
            
                X = st.multiselect("Select Column for X-axis",all_columns_names,key='0033')
            with sca2:
                Y = st.multiselect("Select Column for Y-axis",all_columns_names,key='0044')
            
            X1 = df[X]
            Y1 = df[Y]
            plt.scatter(X1,Y1,s=20,color='coral')
            st.pyplot()  
        #fig = px.scatter(df, x =X,y=Y, color='variety')
        #st.plotly_chart(fig)
        
        #boxplot = all_columns_names.boxplot(column=x, by=y)
        #st.write(plt.boxplot(boxplot))
        #train.boxplot(column='ApplicantIncome', by='Education') 
        #plt.suptitle("") 

#------------ box plot

        if st.sidebar.checkbox("Distribution Plot", key='00332'):
            st.write("Boxplot Plot")
            all_columns_names = df.columns.tolist()       
            X = st.sidebar.multiselect("Select Column",all_columns_names,key='00331')
            X1 = df[X]
            X1.plot.box(figsize=(16,5))
            st.pyplot()
            st.write("Distribution Plot")
            sns.distplot(X1);
            st.pyplot()
 
#---------------------------EDA END----------------------------------       
        
        
#--------------------------------- MODEL BUILDING---------------------------------------        
        
    elif choice == 'Model Building':
        st.subheader("Create ML model")
        
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            st.dataframe(df.head())
            
        if st.sidebar.checkbox("Show Dimensions(Shape)"):
            st.success("In (Row, Column) format")
            st.write(df.shape)
                
        if st.sidebar.checkbox("Data Types"):
            st.success("In (Column Names , Data Type) format")
            st.table(df.dtypes)
            
        if st.sidebar.checkbox("Show Missing Values"):
            st.write(df.isnull().sum())
        
        if st.sidebar.checkbox("Impute Missing Values*"):
            st.info("Currently support ~ dropna | droping the missing values")
            df = df.dropna()
            st.write(df.isnull().sum())
            
        if st.sidebar.checkbox("Remove Columns*"):
            all_columns = df.columns.to_list()
            selected_columns = st.sidebar.multiselect("Select Columns like ID to remove",all_columns)
            #st.write(str(selected_columns))
            new_drop_df = df.drop(selected_columns,axis=1)
            st.write(new_drop_df)
            
        if st.sidebar.checkbox("Define X and Y*"):
            all_columns_names = new_drop_df.columns.tolist()
            #sca1, sca2 = st.beta_columns(2)
            #with sca1:
            if st.sidebar.checkbox("OR Selected X Variables"):
               X = st.sidebar.multiselect("Select Column for Independent Variables X",all_columns_names,key='dx')
               d_x = new_drop_df[X]
               st.info("X-Dependent Variables")
               st.write(d_x)
               
               Y1 = new_drop_df.drop(X,axis=1)              
               st.info("Y-Independent Variable")
               st.write(Y1)
               
            else:
               Y = st.sidebar.multiselect("Select Column for Dependent variable Y*",all_columns_names,key='dy')
               Y1 = new_drop_df[Y]
               st.info("Y-Inependent Variable")
               st.write(Y1)
               
               d_x = new_drop_df.drop(Y,axis = 1)
               st.info("X_Dependent Variables")
               st.write(d_x)
            
           
        #if st.sidebar.checkbox("Transform Categorical variables"):
                #all_columns = df.columns.to_list()
                #selected_columns = st.sidebar.multiselect("Select Columns like ID to transfrm",all_columns,key="trans01")
                
         #   trans_df = pd.get_dummies(d_x)
          #  st.success("Transformation Successful")
           # st.write(trans_df) 
            
            
        if st.sidebar.checkbox("Select Columns to Use*",key='s01'):
            #st.info("Columns to Select")
            all_columns = d_x.columns.to_list()
            if st.sidebar.checkbox("Select X-var Columns",key='s02'):
               selected_columns = st.sidebar.multiselect("Select Columns",all_columns,key='s03')
               trans_df = d_x[selected_columns]
               st.success("X-Var Columns Selected")
               st.dataframe(trans_df)
            
            else:
               st.sidebar.checkbox("Or Select All Columns*",key='s04')
               all_columns = d_x.columns.to_list()
               #st.write(all_columns)
               trans_df = d_x[all_columns]
               st.success("Columns Selected")
               st.dataframe(trans_df)
               
        if st.sidebar.checkbox("Transform Categorical variables*"):
            #all_columns = df.columns.to_list()
            #selected_columns = st.sidebar.multiselect("Select Columns like ID to transfrm",all_columns,key="trans01")
                
            trans_df1 = pd.get_dummies(d_x)
            st.success("Transformation Successful")
            st.write(trans_df1) 
        
        if st.sidebar.checkbox("Scale/Normalize the Data*"):
            sc = StandardScaler()
            scaled_df = sc.fit_transform(trans_df1)
            st.success("Scaled/Normalized Data")
            st.write(scaled_df)
            #X_test = sc.transform(X_test)
            

        if st.sidebar.checkbox("Class Imbalance/Value counts"):
            st.success("Class Imbalance Data for Y Dependent Variable")
            st.write(Y1.value_counts())
            
            
#-----------------------GENERATE AI MODEL--------------------------------        
        
        if st.sidebar.checkbox("Generate Ai Model"):
            X_train,X_test, y_train, y_test = train_test_split(scaled_df, Y1, train_size=0.7, random_state=1)
            algo = ['Logistic Regression','K-Nearest Neighbor(KNN)','Bayes’ Theorem: Naive Bayes','Linear Discriminant Analysis(LDA)','Linear Support Vector Machine(SVM)','Kernel Support Vector Machine(SVM)','SVM: with Paramter Tuning','Decision Tree: Rule-Based Prediction','Random Forest: Ensemble Method','eXtreme Gradient Boosting(XGBoost)']
            st.info("")
            st.success("")
            classifier = st.selectbox('Which algorithm?', algo)
            
            if classifier=='Decision Tree: Rule-Based Prediction':
                dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
                dt_model.fit(X_train, y_train)
                acc = dt_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                
                y_pred_dt = dt_model.predict(X_test)
                cm=confusion_matrix(y_test,y_pred_dt)
                st.write('Confusion matrix: ', cm)
                
                # get importance
                importance = dt_model.feature_importances_
                st.info("Index column:Feature Numbers , Index Values:Score")
                st.write(importance)
                # plot feature importance
                st.success("Important Features Plot")
                plt.bar([x for x in range(len(importance))], importance)
                st.pyplot()
                from sklearn.tree import export_text
                st.info("Rules/Conditions used for prediction ")
                col_names_dt=trans_df1.columns.tolist()
                tree_rules = export_text(dt_model, feature_names=col_names_dt)
                st.write(tree_rules)
                st.info("Visualizing The Tree might be restricted due to HTML length & Width: will be updating soon")
                st.info("")
                st.success("")
                st.warning("")
                
            if classifier=='Random Forest: Ensemble Method':
                 rf1, rf2 = st.beta_columns(2)
                 with rf1:
                    rf_n = st.slider("Select n_estimators/Tree",400,1500,500,key='0rf33')
                    st.write("Default: 500 Trees")
                 with rf2:
                    rf_p = st.select_slider("Select Criterion ",options=["entropy","gini"],value=("entropy"),key='0rf34')
                    st.write("Splitting Criteria:",rf_p)
                 rf_model = RandomForestClassifier(n_estimators = rf_n, criterion = rf_p, random_state = 0)
                 rf_model.fit(X_train, y_train)
                 acc = rf_model.score(X_test, y_test)
                 st.write('Accuracy: ', acc)
                
                 y_pred_rf = rf_model.predict(X_test)
                 cm=confusion_matrix(y_test,y_pred_rf)
                 st.write('Confusion matrix: ',cm)
                
                 # get importance
                 importance = rf_model.feature_importances_
                 st.info("Index column:Feature Numbers , Index Values:Score")
                 st.write(importance)
                 # plot feature importance
                 st.success("Important Features Plot")
                 plt.bar([x for x in range(len(importance))], importance)
                 st.pyplot()
                 st.info("")
                 st.success("")
                 st.warning("")
            
            
            if classifier=='K-Nearest Neighbor(KNN)':
                 knn1, knn2 = st.beta_columns(2)
                 with knn1:
                    k_n = st.slider("Select N_neighbors",3,10,5,key='0k33')
                    st.write("Default metric: Minkowski")
                 with knn2:
                    K_p = st.slider("Select Distance metrics: 1 is equivalent to the Manhattan distance and 2 is equivalent to the Euclidean distance ",1,2,2,key='0k34')
                                
                 knn_model = KNeighborsClassifier(n_neighbors = k_n, metric = 'minkowski', p = K_p)
                 knn_model.fit(X_train, y_train)
                 acc = knn_model.score(X_test, y_test)
                 st.write('Accuracy: ', acc)
                    # Predicting the Test set results
                 y_pred_knn = knn_model.predict(X_test)
                 cm=confusion_matrix(y_test,y_pred_knn)
                 st.write('Confusion matrix: ', cm)
                 st.error("Try another Algorithm for Feature importance")
                 st.info("")
                 st.success("")
                 st.warning("")
                
                
            if classifier == 'Linear Support Vector Machine(SVM)':
                svm_model=SVC(kernel = 'linear', random_state = 0)
                svm_model.fit(X_train, y_train)
                acc = svm_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                svm_pred = svm_model.predict(X_test)
                cm=confusion_matrix(y_test,svm_pred)
                st.write('Confusion matrix: ', cm)
                # get importance
                importance = svm_model.coef_[0]
                st.info("Index column:Feature Numbers , Index Values:Score")
                st.write(importance)
                # plot feature importance
                st.success("Important Features Plot")
                plt.bar([x for x in range(len(importance))], importance)
                st.pyplot()
                st.info("")
                st.success("")
                st.warning("")
                
                
            if classifier == 'Kernel Support Vector Machine(SVM)':
                ksvm_model=SVC(kernel = 'rbf', random_state = 0)
                ksvm_model.fit(X_train, y_train)
                acc = ksvm_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                ksvm_pred = ksvm_model.predict(X_test)
                cm=confusion_matrix(y_test,ksvm_pred)
                st.write('Confusion matrix: ', cm)
                st.error("Try another Algorithm for Feature importance")
                st.info("")
                st.success("")
                st.warning("")
                
            if classifier=='SVM: with Paramter Tuning':
                 svm_o1, svm_o2 = st.beta_columns(2)
                 with svm_o1:
                    svm_gamma = st.select_slider("Select Gamma(gamma) ",options=["scale","auto"],value=("scale"),key='0svm34')
                    st.write("Kernel Gamma:",svm_gamma) 
                    svm_c = st.slider("Select Regularization parameter (c)",0.1,9.9,1.0,key='0svmk33')
                    st.write("Default: 1.0")
                    svm_iter = st.slider("Select Iterations",1,1000,-1,key='0svmk33')
                    st.write("Default: -1 = No limit")
                    
                 with svm_o2:
                    svm_types = st.select_slider("Select Criterion(kernel type) ",options=["poly","rbf","sigmoid","linear"],value=("sigmoid"),key='0svm34')
                    st.write("Kernel Type:",svm_types) 
                    svm_degree = st.slider("Select degree for poly kernel(Ignored by all other Kernels)",2,8,3,key='0svmk33')
                    st.write("Default: 3")
           
                 ksvm_model2=SVC(kernel = svm_types,gamma=svm_gamma,C=svm_c,max_iter = svm_iter,degree=svm_degree,random_state = 0)
                 ksvm_model2.fit(X_train, y_train)                 
                 if st.checkbox("Automate Best Optimal Parameters using Grid Search?"):
                     st.text(" Searching the Best Optimal Parameters, this will take few mins.")
                     st.text("If click please wait until the process is complete else refresh the page")
                     parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                                   {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
                     grid_search = GridSearchCV(estimator = ksvm_model2,param_grid = parameters,scoring = 'accuracy',cv = 10)
                     grid_search = grid_search.fit(X_train, y_train)
                     best_accuracy = grid_search.best_score_
                     best_parameters = grid_search.best_params_
                     st.write("The Best setting will give-",(best_accuracy))
                     st.write(best_parameters)
                     st.info("Re-run the model with the settings above")  
                 else:
                     pass                                                                 
                 acc = ksvm_model2.score(X_test, y_test)
                 st.write('Accuracy: ', acc)
                 ksvm_pred2 = ksvm_model2.predict(X_test)
                 cm=confusion_matrix(y_test,ksvm_pred2)
                 st.write('Confusion matrix: ', cm)
                 st.error("Try another Algorithm for Feature importance")
                 st.info("")
                 st.success("")
                 st.warning("")
                    
                
            if classifier == 'Bayes’ Theorem: Naive Bayes':
                NB_model = GaussianNB()
                NB_model.fit(X_train, y_train)
                acc = NB_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                # Predicting the Test set results
                y_pred_nb = NB_model.predict(X_test)
                cm=confusion_matrix(y_test,y_pred_nb)
                st.write('Confusion matrix: ', cm)

                st.error("Try another Algorithm for Feature importance")
                st.info("")
                st.success("")
                st.warning("")

            if classifier == 'Linear Discriminant Analysis(LDA)':
                lda_model=LinearDiscriminantAnalysis()
                lda_model.fit(X_train, y_train)
                acc = lda_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                lda_pred = lda_model.predict(X_test)
                cm=confusion_matrix(y_test,lda_pred)
                st.write('Confusion matrix: ', cm)
                st.error("Try another Algorithm for Feature importance")
                st.info("")
                st.success("")
                st.warning("")                          
                
                
            if classifier == 'Logistic Regression':
                lr_model= LogisticRegression(random_state = 0)
                lr_model.fit(X_train, y_train)
                acc = lr_model.score(X_test, y_test)
                st.write('Accuracy: ', acc)
                pred_lr = lr_model.predict(X_test)
                cm=confusion_matrix(y_test,pred_lr)
                st.write('Confusion matrix: ', cm)

                # get importance
                importance = lr_model.coef_[0]
                st.info("Index column:Feature Numbers , Index Values:Score")
                st.write(importance)
                # plot feature importance
                st.success("Important Features Plot")
                plt.bar([x for x in range(len(importance))], importance)
                st.pyplot()
                st.info("")
                st.success("")
                st.warning("")
            
                with st.beta_expander("Download The Model"):
                #if we wish to save 
                    #lr_model.save_weights('StackedLSTM.h5')
                    #save the model in the disk
                    import pickle
                    # save the model to disk
                    filename = 'lr_class_model.sav'
                    d=pickle.dump(lr_model, open(filename, 'wb'))
                    
                    href = f'<a href="data:file/txt" download="d">Click here!</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
            if classifier == 'eXtreme Gradient Boosting(XGBoost)':
                 xg_o1, xg_o2 = st.beta_columns(2)
                 with xg_o1:
                    xg_loss = st.select_slider("Select Loss func* ",options=["deviance","exponential"],value=("deviance"),key='0xg34')
                    st.text("deviance=logistic regression,exponential=AdaBoost algorithm")
                    st.write("Selected:",xg_loss)
                    
                    xg_lr = st.slider("Select Learning Rate",0.1,4.9,0.1,key='0xg338')
                    st.write("Default: 0.1")
                    
                    xg_est = st.slider("Select n_estimators, default=100",50,1000,100,key='0xg33n')
                    st.write("Number of boosting stages to perform",xg_est)
                    
                 with xg_o2:
                    xg_crit = st.select_slider("Select Criterion ",options=["friedman_mse","mse","mae"],value=("friedman_mse"),key='0xg3476')
                    st.write("Selected:",xg_crit)
                    
                    xg_feat = st.select_slider("Select max_features ",options=["auto","sqrt","log2","None"],value=("None"),key='0xg3471')
                    st.text("If None,then max_features=n_features.")
                    st.write("Selected:",xg_feat)                   
             
                 xg_model= XGBClassifier(loss=xg_loss,learning_rate=xg_lr, criterion=xg_crit, n_estimators=xg_est, max_features=xg_feat, random_state = 0)
                 xg_model.fit(X_train, y_train)
                 acc = xg_model.score(X_test, y_test)
                 st.write('Accuracy: ', acc)
                 pred_xg = xg_model.predict(X_test)
                 cm=confusion_matrix(y_test,pred_xg)
                 st.write('Confusion matrix: ',cm)
                 st.info("")
                 st.success("")
                 st.warning("")

#-----------------------------------PREDICTION---------------------------------
    
        if st.sidebar.checkbox("Predict"):
            data = st.file_uploader("Upload your data you want to predict",type = ["csv","txt"])
            if data is not None:
                df_temp = pd.read_csv(data,sep=',')
                df_temp = df_temp.dropna()
                st.success("Removed Missing Values")
                st.dataframe(df_temp.head())
                st.info("Mapping the number of the variables from the model")
                st.write("Found columns", len(trans_df.columns))
                
            fr_new_columns = trans_df.columns.to_list()
               #st.write(all_columns)
            trans_newdf = df_temp[fr_new_columns]
            st.success("Columns Mapped")
            st.dataframe(trans_newdf)
            st.write("Mapped columns", len(trans_newdf.columns))
             
            ddd_df1 = pd.get_dummies(trans_newdf)
            st.success("Transformation Successful")
            st.write(ddd_df1) 
                    
            sc = StandardScaler()
            scaled_df1 = sc.fit_transform(ddd_df1)
            st.success("Scaled/Normalized Data")
            st.write(scaled_df1)
            
            #---------------- PREDICT NEW DATA using SELECTED MODEL---------
            
            #st.write(algo) getting the list of model from above
            #algo = ['Logistic Regression','K-Nearest Neighbor(KNN)','Bayes’ Theorem: Naive Bayes','Linear Discriminant Analysis(LDA)','Linear Support Vector Machine(SVM)','Kernel Support Vector Machine(SVM)','SVM: with Paramter Tuning','Decision Tree: Rule-Based Prediction','Random Forest: Ensemble Method','eXtreme Gradient Boosting(XGBoost)']
            selected_model_names = st.selectbox("Select Models To Predict Your New Data",algo)
            st.text(" Note: must use respective Generate Ai model before predicting")
            if selected_model_names =='Logistic Regression':
                results = lr_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)
                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='K-Nearest Neighbor(KNN)':
                results = knn_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
            
            elif selected_model_names=='Bayes’ Theorem: Naive Bayes':
                results = NB_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='Linear Discriminant Analysis(LDA)':
                results = lda_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='Linear Support Vector Machine(SVM)':
                results = svm_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable column as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='Kernel Support Vector Machine(SVM)':
                results = ksvm_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='SVM: with Paramter Tuning':
                results = ksvm_model2.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='Decision Tree: Rule-Based Prediction':
                results = dt_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                
                 #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='Random Forest: Ensemble Method':
                results = rf_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
                
            elif selected_model_names=='eXtreme Gradient Boosting(XGBoost)':
                results = xg_model.predict(scaled_df1)
                st.success("Predicted Results")
                st.write(results)
                 
            #------------Putting All Together The Results--------------------------
                inv_df = sc.inverse_transform(scaled_df1)

                #need to convert to pandas else will throw error 'col' not found as it a numpy object
                inv_df2  = pd.DataFrame(inv_df)
                #st.write(inv_df2)
                       
                #geting the column name as list
                c_names = ddd_df1.columns.tolist()
            
                #adding column names to the dataset                        
                inv_df2.columns=c_names
                #st.write(inv_df2)
                st.success("Putting all together your uploaded data and the predicted results")
            
                #concat--------
                #st.write(Y1.columns)
                #need to convert to pandas else numpy cant find columns error
                y_col = pd.DataFrame(Y1.columns)
                #st.write(y_col)
                #getting the column name as list
                y_names = y_col.values.tolist()
                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error
                results2=pd.DataFrame(results)
                #renaming/labeling the Y column          
                results2.columns=y_names
                #st.write(results2)
       
                #series not applicable columna as name s = pd.Series(results2)
                p_results = pd.concat([inv_df2,results2],axis=1)
                #working:p_results = pd.concat([inv_df2,s],axis=1)
                st.write(p_results)
       #--------------- THE END---------------------------     
         
                                
        
    elif choice == 'About':
        st.subheader("About")
        st.text("Thank you for your time")
        
        st.markdown("""
Hi I’m Bob aka. Rupak Roy. Things i write about frequently on Quora & Linkedin: analytics For Beginners, Data Science, Machine Learning, Deep learning, Natural Language Processing (NLP), Computer Vision, Big Data Technologies, Internet Of Thins and many other random topics of interest.
I formerly Co-founded various Ai based projects to inspire and nurture the human spirit with the Ai training on how to leverage on how to leverage Ai to solve problems for an exponential growth. IF you need such app just ping, happy to help.

My Career Contour consists of various technologies starting from Masters of Science in Information Technology to Commerce with the privilege to be Wiley certified in various Analytical Domain.
My alternative internet presences, Facebook, Blogger, Linkedin, Medium, Instagram, ISSUU and with Data2Dimensions

If you wish to learn more about Data Science follow me at:

~ Medium [@rupak.roy](https://medium.com/@rupak.roy)

~ Linkedin [@bobrupak](https://www.linkedin.com/in/bobrupak/)

My Fav. Quote:

Millions saw the apple fall but only Newton asked why! ~ “Curiosity is the spark of perfection and innovations. So connect with data and discover sync“
""")
        st.image('img/prism.gif')
                
        with st.beta_expander("Suprise!"):
            st.title("COLLECT YOUR FULL VERSION NLP APP @ ping_me #socialmedia")
            st.image('img/office.jpg')
            st.info("")
            st.success("")
            st.warning("")
            st.error("")
        
        
        
        

if __name__ == '__main__':
    main()