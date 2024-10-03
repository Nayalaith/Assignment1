import streamlit as st
import requests

st.title("Movie Recommendation System")

genres = st.text_input("Enter genres (space-separated)")
emotions = st.text_input("Enter emotions (space-separated)")
length = st.selectbox("Select length", ['short (45-75 min)', 'medium (75-120 min)', 'long (120-200 min)'])

if st.button("Recommend Movies"):
    data = {
        "genres": genres,
        "emotions": emotions,
        "length": length
    }
    response = requests.post("http://api:8000/recommend/", json=data)
    recommendations = response.json().get("recommendations", [])
    st.write("Top Movie Recommendations:")
    for movie in recommendations:
        st.write(movie)

