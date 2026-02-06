# shepherd_pro/app/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Shepherd RL Hub", 
    page_icon="🐑", 
    layout="wide"
)

# --- Header Section ---
st.title("🐑 Shepherd RL Project")
st.markdown("### Autonomous Agent Training & Demonstration Hub")
st.markdown("---")

# --- Introduction ---
st.markdown("""
Welcome! This application serves as the control center for your Reinforcement Learning Shepherd agents.
Navigate using the **sidebar** or the cards below.
""")

st.write("") # Spacer

# --- Navigation Cards (UI Only) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 🏋️ Training")
    st.write("Train PPO, TD3, or DQN agents in the Shepherd environment. Monitor progress via TensorBoard.")
    st.markdown("*Go to sidebar → Training*")

with col2:
    st.success("### 🎮 Demo")
    st.write("Visualize your trained agents in real-time. Watch how they herd sheep and avoid obstacles.")
    st.markdown("*Go to sidebar → Demo*")

with col3:
    st.warning("### 🧪 Test")
    st.write("Run specific custom scenarios and stress-tests on your agents to evaluate edge cases.")
    st.markdown("*Go to sidebar → Test*")

st.markdown("---")
st.caption("Developed for the Shepherd RL Project. Select a page from the sidebar to begin.")