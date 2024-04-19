#Please see report - this is not the main submission

import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

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
    <h1 style="color: #00A36C;">User Recipe Recommendation System Application</h1>
    </div>
    <h6 style="color: #00A36C;">This system works by recommending recipes based on user ID and previously rated recipes.</h6>
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
user_input = st.text_input("Enter user ID:")


# Load the dataset
raw_recipes = pd.read_csv("RAW_recipes copy.csv")

# Keep only relevant columns
recipes = raw_recipes[['recipe_id', 'name', 'steps', 'ingredients', 'n_ingredients']]

# Load interactions data
interactions = pd.read_csv("RAW_interactions.csv")

# Define the Reader object
reader = Reader(rating_scale=(1, 5))

# Load the dataset for Surprise
data = Dataset.load_from_df(interactions[['user_id', 'recipe_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD algorithm
algo = SVD()

# Train the model
algo.fit(trainset)

# Get top N recommendations for a user
def get_top_n_recommendations(user_id, n=10):
    # Get a list of all recipe IDs
    all_recipe_ids = recipes['recipe_id'].unique()
    
    # Remove the recipe IDs the user has already interacted with
    interacted_recipes = interactions[interactions['user_id'] == user_id]['recipe_id']
    remaining_recipe_ids = [recipe_id for recipe_id in all_recipe_ids if recipe_id not in interacted_recipes]
    
    # Predict ratings for remaining recipes
    predictions = [algo.predict(user_id, recipe_id) for recipe_id in remaining_recipe_ids]
    
    # Sort predictions by estimated rating in descending order
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    # Get top N recommendations
    top_n_recommendations = top_predictions[:n]
    
    # Extract recipe IDs from recommendations
    top_n_recipe_ids = [prediction.iid for prediction in top_n_recommendations]
    
    # Get recipe names for top recommendations
    top_n_recipe_names = recipes[recipes['recipe_id'].isin(top_n_recipe_ids)]['name']
    
    return top_n_recipe_names

#Get top 10 recommendations for user with ID 1
# user_id = 1
# top_recommendations = get_top_n_recommendations(user_id)
# print("Top 10 recommended recipes for user", user_id, ":\n", top_recommendations)

if user_input:
    user_id = int(user_input)
    top_recommendations = get_top_n_recommendations(user_id)
    st.write("Top 10 recommended recipes for user", user_id, ":\n", top_recommendations)