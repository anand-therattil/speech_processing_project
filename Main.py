import streamlit as st

page_1 = st.Page("page_1.py",title="Audio Analytics",icon=":material/bar_chart:")
page_2 = st.Page("page_2.py",title="Text Analytics", icon=":material/bar_chart:")
pg = st.navigation([page_1, page_2])
pg.run()

