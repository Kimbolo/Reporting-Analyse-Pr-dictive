import streamlit as st
import os

def render_sidebar():
    LOGO_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "logo.jpg"
    )

    st.sidebar.image(LOGO_PATH, width=180)

    st.sidebar.markdown(
        """
        <div style="text-align:center; font-weight:600; font-size:14px;">
            N'NAM AGRO INDUSTRIE<br/>
            <span style="font-size:11px; color:gray;">
                Pilotage & Performance
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")