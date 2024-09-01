import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

@st.cache_data # Decorator to prevent data from being loaded for every step
def load_data():
    # Unzip the file
    with zipfile.ZipFile("survey_results_public.zip", "r") as zip_ref:
        zip_ref.extractall("survey_results_public")

    # Check if the folder exists
    if os.path.exists("survey_results_public"):
        # Access the folder and load the CSV file into a DataFrame
        csv_file_path = os.path.join("survey_results_public", "survey_results_public.csv")
        df = pd.read_csv("survey_results_public.csv")
        
        # Filter and process the DataFrame as before
        df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
        df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
        df = df[df["Salary"].notnull()]
        df = df.dropna()
        df = df[df['Employment'].str.contains('Employed, full-time')]
        df = df.drop("Employment", axis=1)
        country_map = shorten_categories(df.Country.value_counts(), 400)
        df["Country"] = df["Country"].map(country_map)
        df = df[df["Salary"] <= 300000]
        df = df[df["Salary"] >= 15000]
        df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
        df["EdLevel"] = df["EdLevel"].apply(clean_education)
        return df
    else:
        print("The folder survey_results_public does not exist.")
        return None

df = load_data()

def show_explore_page():
   st.title("Explore Developers Salaries")

   st.write(
       """ ### Stack Overflow Developer Survey 2023
    """
   )
   
   data = df["Country"].value_counts()

   fig1, ax1 = plt.subplots()
   ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90, labeldistance=1.1)
   ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
   plt.xticks(rotation=45)
   st.write("""#### Survey Participants""")
   
   st.pyplot(fig1)
   st.write(
        """
    #### Mean Salary Based On Country
    """
    )
   
   data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
   
   st.bar_chart(data)
   st.write(
        """
    #### Mean Salary Based On Experience
    """
    )
   
   data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
   st.line_chart(data)
