import streamlit as st

conn = st.connection("course_assistant")
df = conn.query("SELECT * FROM conversations LIMIT 20")
st.dataframe(df)