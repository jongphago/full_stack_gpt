import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")
    
tab1, tab2, tab3 = st.tabs(["tab1", "tab2", "tab3"])

with tab1:
    st.write("tab1")
    
with tab2:
    st.write("tab2")
    
with tab3: 
    st.write("tab3")