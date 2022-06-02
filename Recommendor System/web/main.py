import streamlit as st
import pickle 

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('Movie Recommender System')
recommended_movies = []


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


option = st.selectbox(
    "Select a movie to show recommendations",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(option)
    st.write('Selected Movie: ', option)
    st.write('Recommendations')
    for recommendation in recommendations:
        st.write(recommendation)
