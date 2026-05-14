import matplotlib
matplotlib.use('Agg')
from sim.environment import SmartIrrigationEnv
from sim.q_learning_agent import QLearningAgent

import pickle
import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# LOAD TRAINED SARSA MODEL
# ======================================

agent = QLearningAgent()

with open("models/sarsa_tuned_v1_qtable.pkl", "rb") as file:

    agent.q_table = pickle.load(file)


# ======================================
# CREATE ENVIRONMENT
# ======================================

env = SmartIrrigationEnv()


# ======================================
# SETTINGS
# ======================================

episodes = 100

steps_per_episode = 20


# ======================================
# METRICS
# ======================================

baseline_rewards = []
rl_rewards = []

baseline_water = []
rl_water = []


# ======================================
# BASELINE POLICY
# FIXED TIMER IRRIGATION
# ======================================

for episode in range(episodes):

    env.reset()

    total_reward = 0
    total_water = 0

    for step in range(steps_per_episode):

        # fixed timer logic
        # irrigate every alternate step

        if step % 2 == 0:

            action = 1

        else:

            action = 0

        next_state, reward, done, water_used = env.step(action)

        total_reward += reward
        total_water += water_used

    baseline_rewards.append(total_reward)
    baseline_water.append(total_water)


# ======================================
# RL POLICY
# ======================================

for episode in range(episodes):

    state = env.reset()

    total_reward = 0
    total_water = 0

    for step in range(steps_per_episode):

        action = agent.choose_action(state)

        next_state, reward, done, water_used = env.step(action)

        state = next_state

        total_reward += reward
        total_water += water_used

    rl_rewards.append(total_reward)
    rl_water.append(total_water)


# ======================================
# RESULTS
# ======================================

results = {

    "Metric": [
        "Average Reward",
        "Average Water Usage"
    ],

    "Baseline": [

        sum(baseline_rewards) / len(baseline_rewards),

        sum(baseline_water) / len(baseline_water)
    ],

    "RL_Policy": [

        sum(rl_rewards) / len(rl_rewards),

        sum(rl_water) / len(rl_water)
    ]
}


results_df = pd.DataFrame(results)

print("\n===== BASELINE VS RL RESULTS =====\n")

print(results_df)


# ======================================
# SAVE CSV
# ======================================

results_df.to_csv(
    "experiments/baseline_vs_rl.csv",
    index=False
)


# ======================================
# PLOT
# ======================================

plt.figure(figsize=(10, 5))

plt.plot(
    baseline_rewards,
    label="Baseline Policy"
)

plt.plot(
    rl_rewards,
    label="SARSA RL Policy"
)

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.title(
    "Baseline vs RL Policy"
)

plt.legend()

plt.savefig(
    "plots/baseline_vs_rl_plot.png"
)

