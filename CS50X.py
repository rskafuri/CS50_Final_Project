import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.sidebar.header('Chose you file')
raw_file = st.sidebar.file_uploader(
    label='You are welcome to chose one of our examples',
    help='Make sure your file is in UTF-8 encoding. Check exmples for more information',
    type=['.csv'])

st.sidebar.markdown("""
                    ***
                    Download sample files at [Kaggle](https://www.kaggle.com/datasets)
                    """,unsafe_allow_html=True)


st.markdown("""
# AutoML for Beginners

Upload your **.csv** file and make use of simple Machine Learning features.

In this app you will:

1. Upload your dataset
2. Plot pairwise relationships of your dataset features
3. Choose a feature as your prediction target
4. Use a Random Forest Regressor model to determine feature importance

###### by: Roberto Sebba Kafuri

***

""")

if raw_file is not None:
    try:
        df = pd.read_csv(raw_file)
        st.write(df)
    except:
        st.markdown("That didn't go so well. Try another file")
    
    st.markdown("""
    ***
    You will learn that most machine learning models don't do well with columns
    containing **categorical variables**. They prefer numbers instead!
    
    They also can't deal very well with rows containing **missing values**.
    
    There are ways we can deal with both those things, but for this tutorial we
    will simply **remove those rows and columns** from our dataset.
    
    Let's continue your ML journey by cleaning yout data
    
    """)
    
    df = df.dropna()
    df = df.select_dtypes(exclude=['object'])
    st.write(df)
    
    st.markdown("""
    ***
    Sometimes it is hard to make sense of all the values on a data frame. That is
    why it is useful to make an Exploratory Data Analisys.
    
    During this analisys we will plot graphs that help us understand the
    **relationships between features** of your dataset.
    
    Lets see how our variables interact
    
    """)
    
    st.pyplot(sns.pairplot(data=df))
    
    st.markdown("""
    ***
    See any interesting patters?
    
    Its's time to start our machine learning model. Today we will use the
    **Random Forest Generator Model**, a type of **supervised learning** model.
    
    This means we should choose the variable we are going to predict, based on
    all other variables in our dataset.
    
    Choose your target between your dataset columns
    
    """)
    
    target = st.selectbox(label='Target', options=df.columns)
    
    df_X = df.copy()
    df_y = df_X.pop(target)
    
    model = RandomForestRegressor(random_state=0)
    
    model.fit(df_X, df_y)
    
    # Create arrays from feature importance and feature names
    feature_importance = np.array(model.feature_importances_)
    feature_names = np.array(df_X.columns)
    
    # Create a DataFrame using a Dictionary
    data = {'FEATURES': feature_names, 'IMPORTANCE': feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['IMPORTANCE'], ascending=False, inplace=True)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax = sns.barplot(data=fi_df, x='IMPORTANCE', y='FEATURES')
    
    st.pyplot(fig)
    
    st.markdown("Try multiple targets and see what happens!")
