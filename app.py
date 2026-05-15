import streamlit as st
import requests
import pandas as pd
from PIL import Image


# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="🌱",
    layout="wide"
)


# ======================================
# TITLE
# ======================================

st.title("🌱 Smart Irrigation using Reinforcement Learning")

st.caption(
    "An Intelligent Reinforcement Learning + MLOps based Smart Irrigation Optimization System"
)

st.markdown("""
This dashboard demonstrates a complete Reinforcement Learning based
Smart Irrigation System integrated with modern MLOps tools including:

- FastAPI
- MLflow
- Docker
- GitHub Actions
- Streamlit
- Monitoring and Logging

The system predicts whether irrigation should occur based on
environmental conditions such as soil moisture, temperature, and rainfall.
""")

st.markdown("---")


# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("Input Parameters")

st.sidebar.markdown("""
### Environmental State Levels

The Reinforcement Learning agent works on discretized
environmental state values.

| Value Range | Meaning |
|---|---|
| 0 – 3 | Low |
| 4 – 7 | Medium |
| 8 – 10 | High |

---

### Parameter Descriptions

#### Soil Moisture
- Low → Dry soil
- High → Wet soil

#### Temperature
- Low → Cool climate
- High → Hot climate

#### Rainfall
- Low → No rainfall
- High → Heavy rainfall
""")

soil_moisture = st.sidebar.slider(
    "Soil Moisture",
    0,
    10,
    2
)

temperature = st.sidebar.slider(
    "Temperature",
    0,
    10,
    5
)

rainfall = st.sidebar.slider(
    "Rainfall",
    0,
    10,
    1
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Best Performing Algorithm: SARSA"
)


# ======================================
# PROJECT METRICS
# ======================================

st.subheader("Project Metrics")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Best Algorithm",
    "SARSA"
)

metric2.metric(
    "Average Reward",
    "71.58"
)

metric3.metric(
    "Deployment",
    "Docker + FastAPI"
)

st.markdown("---")


# ======================================
# FASTAPI PREDICTION
# ======================================

st.subheader("Real-Time Irrigation Prediction")

st.markdown("""
The dashboard sends environmental inputs to the deployed FastAPI model.
The RL agent then predicts whether irrigation should occur.
""")

if st.button("Predict Irrigation Decision"):

    url = "http://127.0.0.1:8000/predict"

    payload = {
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "rainfall": rainfall
    }

    try:

        response = requests.post(
            url,
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            if result["decision"] == "Irrigate":

                st.success(
                    f"Prediction: {result['decision']}"
                )

            else:

                st.warning(
                    f"Prediction: {result['decision']}"
                )

            st.json(result)

        else:

            st.error(
                "FastAPI returned an error."
            )

    except:

        st.error(
            "FastAPI server is not running."
        )

st.markdown("---")


# ======================================
# PROJECT OVERVIEW
# ======================================

st.subheader("Project Overview")

st.markdown("""

### Objective

The objective of this project is to optimize irrigation decisions
using Reinforcement Learning in order to:

- Reduce water wastage
- Improve irrigation efficiency
- Automate irrigation decisions
- Support sustainable agriculture

---

### Algorithms Used

- Q-Learning
- SARSA
- Deep Q Network (DQN)

---

### Best Performing Model

SARSA achieved the best overall performance after
hyperparameter tuning and evaluation.

---

### MLOps Components Implemented

- MLflow Experiment Tracking
- FastAPI Deployment
- Docker Containerization
- Monitoring and Logging
- GitHub Version Control
- Streamlit Dashboard
""")

st.markdown("---")


# ======================================
# TRAINING PLOTS
# ======================================

st.subheader("Training and Evaluation Plots")

col1, col2 = st.columns(2)

# ======================================
# BASELINE VS RL
# ======================================

with col1:

    st.markdown("### Baseline vs Reinforcement Learning")

    image1 = Image.open(
        "plots/baseline_vs_rl_plot.png"
    )

    st.image(
        image1,
        use_container_width=True
    )

    st.info("""
The Reinforcement Learning based irrigation policy
consistently achieves higher rewards compared to
the traditional baseline irrigation strategy.
""")

# ======================================
# Q LEARNING
# ======================================

with col2:

    st.markdown("### Q-Learning Reward Plot")

    image2 = Image.open(
        "plots/ql_reward_plot.png"
    )

    st.image(
        image2,
        use_container_width=True
    )

    st.info("""
Q-Learning improves its irrigation policy through
continuous interaction with the environment.
Rewards gradually improve during training.
""")


# ======================================
# SECOND ROW
# ======================================

col3, col4 = st.columns(2)

# ======================================
# SARSA
# ======================================

with col3:

    st.markdown("### SARSA Reward Plot")

    image3 = Image.open(
        "plots/sarsa_reward_plot.png"
    )

    st.image(
        image3,
        use_container_width=True
    )

    st.success("""
SARSA achieved the best overall performance and
policy stability among all evaluated algorithms.
""")

# ======================================
# DQN
# ======================================

with col4:

    st.markdown("### DQN Reward Plot")

    image4 = Image.open(
        "plots/dqn_reward_plot.png"
    )

    st.image(
        image4,
        use_container_width=True
    )

    st.info("""
Deep Q Networks use neural networks for value approximation.
However, SARSA performed better for this irrigation environment.
""")


# ======================================
# FINAL COMPARISON
# ======================================

st.subheader("Final Algorithm Comparison")

image5 = Image.open(
    "plots/final_algorithm_comparison.png"
)

st.image(
    image5,
    use_container_width=True
)

st.success("""
Final comparison results show SARSA outperforming
Q-Learning and DQN in terms of average reward,
policy consistency, and irrigation efficiency.
""")

st.markdown("---")


# ======================================
# RESULTS TABLE
# ======================================

st.subheader("Evaluation Results")

results_df = pd.DataFrame({

    "Algorithm": [
        "Q-Learning",
        "SARSA",
        "DQN"
    ],

    "Average Reward": [
        70.34,
        71.58,
        45.21
    ],

    "Performance": [
        "Good",
        "Best",
        "Moderate"
    ]
})

st.dataframe(
    results_df,
    use_container_width=True
)

st.markdown("---")


# ======================================
# FOOTER
# ======================================

st.markdown("""
### Conclusion

This project successfully demonstrates the integration of
Reinforcement Learning with MLOps practices for intelligent
and automated irrigation optimization.

The system combines:
- RL based decision making
- Experiment tracking
- API deployment
- Docker containerization
- Monitoring
- Interactive dashboard visualization

to create a complete end-to-end MLOps pipeline.
""")

st.markdown("---")

st.caption(
    "Smart Irrigation RL + MLOps Dashboard"
)