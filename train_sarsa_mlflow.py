from sim.environment import SmartIrrigationEnv
from sim.sarsa_agent import SARSAAgent

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import mlflow


# ======================================
# EXPERIMENT CONFIG
# ======================================

experiment_name = "SARSA_MLflow"

learning_rate = 0.05
discount_factor = 0.99
epsilon = 0.10

episodes = 1000
steps_per_episode = 20


# ======================================
# SET MLFLOW EXPERIMENT
# ======================================

mlflow.set_experiment(
    experiment_name
)


# ======================================
# START MLFLOW RUN
# ======================================

with mlflow.start_run():

    # log hyperparameters

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    mlflow.log_param(
        "discount_factor",
        discount_factor
    )

    mlflow.log_param(
        "epsilon",
        epsilon
    )

    mlflow.log_param(
        "episodes",
        episodes
    )


    # create environment

    env = SmartIrrigationEnv()

    agent = SARSAAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon
    )


    # training

    episode_rewards = []

    for episode in range(episodes):

        state = env.reset()

        action = agent.choose_action(state)

        total_reward = 0

        for step in range(steps_per_episode):

            next_state, reward, done, water_used = env.step(action)

            next_action = agent.choose_action(next_state)

            agent.update_q_table(
    state,
    action,
    reward,
    next_state,
    next_action
)

            state = next_state
            action = next_action

            total_reward += reward

        episode_rewards.append(total_reward)
        mlflow.log_metric(
    "episode_reward",
    total_reward,
    step=episode
)


    # final metrics

    average_reward = sum(
        episode_rewards
    ) / len(episode_rewards)

    max_reward = max(episode_rewards)


    # log metrics

    mlflow.log_metric(
        "average_reward",
        average_reward
    )

    mlflow.log_metric(
        "max_reward",
        max_reward
    )


    # save model

    model_path = "sarsa_mlflow_qtable.pkl"

    with open(model_path, "wb") as file:

        pickle.dump(agent.q_table, file)


    mlflow.log_artifact(model_path)


    # generate plot

    plt.figure(figsize=(12, 6))

    plt.plot(
        episode_rewards,
        alpha=0.4,
        label="Raw Rewards"
    )

    moving_avg = pd.Series(
        episode_rewards
    ).rolling(window=50).mean()

    plt.plot(
        moving_avg,
        linewidth=3,
        label="Moving Average"
    )

    plt.title(
        "SARSA Reward Curve"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.legend()

    plot_path = "sarsa_mlflow_plot.png"

    plt.savefig(plot_path)

    mlflow.log_artifact(plot_path)

    plt.close()


    print("\nMLflow experiment completed.")