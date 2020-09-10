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
#import plotly.express as px

#ML pckgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#disable warning message
st.set_option('deprecation.showfileUploaderEncoding', False)

 

def main():
    """Auto Machine Learning app with Streamlit """
    
    st.title("Machine Learning Classifier App")
    st.text( "Using STreamlit ==0.66+ upload data with only numerical values ")
    st.text("~ Currently accepting comma seperated file")
    st.write("""
             Sample Dataset [@rupak-roy Github](https://github.com/rupak-roy/dataset-streamlit) in csv by Bob.
             """)
    activities = ["EDA","Plot","Model Building","About"]
    
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            st.dataframe(df.head())
            
            if st.checkbox("Show shape"):
                st.write(df.shape)
           
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)           
            
            if st.checkbox("Select Columns to Show"):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns ]
                st.dataframe(new_df)
            if st.checkbox("Show summary"):
                st.write(df.describe())
            if st.checkbox("Show Value Counts "):
               st.write(df.iloc[:,-1].value_counts())        
            if st.checkbox("Corelation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()    
        
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot= st.selectbox("Select 1 Columns", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot) 
                st.pyplot()
        
        
    elif choice == 'Plot':
        st.subheader("Data Visualization")
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            st.dataframe(df.head())
        
        if st.checkbox("Corelation with Seaborn"):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()    
        
        if st.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            columns_to_plot= st.selectbox("Select 1 Columns", all_columns)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot) 
            st.pyplot()
        
    
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select type fo plot",["Area","Bar","Line","Hist","Scatter","Box","kde"])
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
            
        elif type_of_plot =='Hist':
            cust_data = df[selected_columns_names]
            plt.hist(cust_data)
            st.pyplot()
            
            
        #Custom Plot
        elif type_of_plot:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()
        
        
    elif choice == 'Model Building':
        st.subheader("Building ML model")
        
        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
        if data is not None:
            df = pd.read_csv(data,sep=',')
            df = df.dropna()
            st.dataframe(df.head())
            
            #model Building
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7
            
            # Model 
            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))
            
            #evaluate each in turn
            
            #list
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            
            for name,model in models:
                kfold = model_selection.KFold(n_splits = 10,random_state = seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                all_models.append(accuracy_results)
                
            if st.checkbox("Metrics as table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean, model_std),columns =["Model Name","Model Accuarcy","Standard Deviation"]))
            
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)
        
        
        
    elif choice == 'About':
        st.subheader("About")
        st.text("Thank you for your time")
        
        st.markdown("""
Hi I’m Bob aka. Rupak Roy. Things i write about frequently on Quora & Linkedin: analytics For Beginners, Data Science, Machine Learning, Deep learning, Natural Language Processing (NLP), Computer Vision, Big Data Technologies, Internet Of Thins and many other random topics of interest.
I formerly Co-founded various Ai based projects to inspire and nurture the human spirit with the Ai training on how to leverage on how to leverage Ai to solve problems for an exponential growth.

My Career Contour consists of various technologies starting from Masters of Science in Information Technology to Commerce with the privilege to be Wiley certified in various Analytical Domain.
My alternative internet presences, Facebook, Blogger, Linkedin, Medium, Instagram, ISSUU and my very own Data2Dimensions

If you wish to learn more about Data Science follow me at:

~ Medium [@rupak.roy](https://medium.com/@rupak.roy)

~ Linkedin [@bobrupak](https://www.linkedin.com/in/bobrupak/)

My Fav. Quote:

Millions saw the apple fall but only Newton asked why! ~ “Curiosity is the spark of perfection and innovations. So connect with data and discover sync“
""")
        
        
        
        

if __name__ == '__main__':
    main()