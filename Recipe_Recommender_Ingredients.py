#this is the main submission file

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st

# Define the HTML template for the front end with custom styles
html_temp = """
    <style>
    body {
        background-color: #0047AB
        font-family: Helvetica, sans-serif;
    }
    h1 {
        color: #1E56A0;  
    }
    </style>
    <div style="background-color:#0047AB;text-align:center;">
    <h1 style="color: #00A36C;">Recipe Recommendation System Application</h1>
    </div>
    <h6 style="color: #00A36C;">This system works by recommending recipes based on the ingredients you provide, so take a look around your kitchen and see what you have. The more ingredients you provide, the better the recommendations</h6>
"""

# Set the page configuration
st.set_page_config(
    page_title="Recipe Recommender",
    initial_sidebar_state="auto",
    page_icon=None,
    menu_items=None,
    layout="centered",
)


# Display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)

# Get user input
user_input = st.text_input("Enter ingredients separated by commas:")


# Load the data
df = pd.read_csv('RAW_recipes.csv')
df = df.drop_duplicates('name')
df = df.dropna()

#ingredients starts with [ and ends with ] , are surrounded in '' and separated by ,
#clean all this up. same with steps

df['ingredients'] = df['ingredients'].str.replace('[', '')
df['ingredients'] = df['ingredients'].str.replace(']', '')
df['ingredients'] = df['ingredients'].str.replace("'", '')

#steps also need to be cleaned up. Unlike ingredients, step separation is difficult because many steps have commas within them we do not want to remove
#so instead of just cleaning the commas that separate the steps, we will replace the ', ' between steps with a * and then split the steps by *
df['steps'] = df['steps'].str.replace('[', '')
df['steps'] = df['steps'].str.replace(']', '')
df['steps'] = df['steps'].str.replace("', '", '*')
df['steps'] = df['steps'].str.replace("'", '')

#drop the columns we don't necessarily need
df = df.drop('id', axis=1)
df = df.drop('n_ingredients', axis=1)
df = df.drop('contributor_id', axis=1)
df = df.drop('submitted', axis=1)
df = df.drop('tags', axis=1)
df = df.drop('nutrition', axis=1)
df = df.drop('n_steps', axis=1)

#function to recommend dishes based on user input
def recommend_dishes(data, user_input):

    #convert user input to lowercase as all ingredients are lowercase
    user_input = user_input.lower()

    #create a count vectorizer object
    vectorizer = TfidfVectorizer()
    #fit and transform the data
    tfidf_matrix = vectorizer.fit_transform(data['ingredients'])
    #transform the user input
    user_vector = vectorizer.transform([user_input])
    #calculate the cosine similarity
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix)
    
    #originally used count vectorizer but switched to tfidf vectorizer as it gave better results
    
    #tfidf vectorizer results
    #testing = "chicken, rice, garlic, onion, cheese"
    #0.2 gives 13720 recipes
    #0.3 gives 2158 recipes
    #0.4 gives 251 recipes!     

    
    #count vectorizer results
    #testing = "chicken, rice, garlic, onion, cheese"
    #0.2 breaks the web app
    #0.3 gives 17740 recipes
    #0.4 gives 3511 recipes

    #get the indices of the recipes that have a cosine similarity greater than 0.4 - this number was chosen after some testing
    matches = [(index, row) for index, row in enumerate(cosine_sim[0]) if row > 0.4]
    #get the recommended dishes
    recommended_dishes = data.iloc[[index for index, _ in matches]]
    #return the recommended dishes
    return recommended_dishes


# Quick test below
# user_input = "chicken, rice, garlic, onion, cheese"

# for index, row in recommend_dishes(df, user_input).iterrows():
#     print(row['name'], "\n", row['ingredients'], "\n", row['steps'])


# Display the recommended dishes
if user_input:
    recommended_dishes = recommend_dishes(df, user_input)

    if recommended_dishes.empty:
        st.write("No recipes found with the ingredients provided. Please try again with different or fewer ingredients.")
    else:
        st.write("Found the following ", recommended_dishes.shape[0], " recipes for you")
        st.write("\n\n")
        i = 1
        for index, row in recommended_dishes.iterrows(): 

            #use an expander because it looks better
            name = row['name'].upper()
            with st.expander(f"Recipe {i}: {name}", expanded=False):
                st.write("Recipe Name: ")
                #write the name in all caps
                st.write(row['name'].upper())
                st.write("Ingredients: ")
                ingredients = row['ingredients'].split(',')
                st.markdown('\n'.join([f"- {ingredient}" for ingredient in ingredients]))
                st.write("Steps: ")
                steps = row['steps'].split('*')
                st.markdown('\n'.join([f"- {step}" for step in steps]))
                st.write("\n\n")
                i += 1
            