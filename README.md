# AutoML for Beginners

When I began to learn how to program my end-goal was always to get into machine learning, as I found it fascinating.
This app's goal is to teach anyone how simple Machine Learning can be.

I wanted to make it so the whole application would run as soon as the user provided the desired DataFrame. I also
strived to teach what the algorithm as doing in the background, so that the user wouldn't get lost during the process.

The first step for using the app is to upload a **.csv** file. If the user hasn't any, I also linked the Kaggle
website, which contains multiple interesting DataFrames that can be download and explored.

```
if raw_file is not None:
    try:
        df = pd.read_csv(raw_file)
        st.write(df)
    except:
        st.markdown("That didn't go so well. Try another file")
```


After uploading the file, the algorithm will automatically perform the necessary Data Cleaning. Removing all
rows with missing values and all categorical variable columns may not be the best way to clean a DataFrame, but that
way I ensured it would work with most, if not all, files.

```
df = df.dropna()
df = df.select_dtypes(exclude=['object'])
```

The next step was to perform a simple EDA (Exploratory Data Analysis). To do that I chose to use the seaborn library
to display the pairwise relationship scatterplots of all features. Again, in a customized setting this would probably
not suffice, but it was generic enough to fit most, if not all, files.

`st.pyplot(sns.pairplot(data=df))`

The last step was to run a **Random Forest Regressor** model in our DataFrame and display the feature importance.
To run the model, the user is required to choose a target through a dropdown menu, the model will automatically rerun
if the target is changed by the user.

```
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
```
