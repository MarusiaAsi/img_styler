import streamlit as st

col11, col22, col33 = st.columns(3)
with col11:
    st.caption("Картинка")
    st.image("catick.jpg")
    st.image("catick.jpg")
    st.image("catick.jpg")

with col22:
    st.caption("Стиль")
    st.image("style.jpg")
    st.image("style2.jpg")
    st.image("style3.jpg")

with col33:
    st.caption("Итог")
    st.image("total.jpg")
    st.image("total2.png")
    st.image("total3.png")
