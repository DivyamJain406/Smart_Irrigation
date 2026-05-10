from fastapi import FastAPI
from pydantic import BaseModel
import logging

import pickle

from sim.q_learning_agent import QLearningAgent


logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ======================================
# LOAD TRAINED MODEL
# ======================================

model_path = "models/q_learning_tuned_v1_qtable.pkl"

agent = QLearningAgent()

with open(model_path, "rb") as file:

    agent.q_table = pickle.load(file)


# ======================================
# CREATE FASTAPI APP
# ======================================

app = FastAPI()


# ======================================
# INPUT SCHEMA
# ======================================

class IrrigationInput(BaseModel):

    soil_moisture: int
    temperature: int
    rainfall: int


# ======================================
# PREDICTION ROUTE
# ======================================

@app.post("/predict")

def predict(data: IrrigationInput):

    state = (
        data.soil_moisture,
        data.temperature,
        data.rainfall
    )

    action = agent.choose_action(state)

    if action == 1:
        decision = "Irrigate"

    else:
        decision = "Do Not Irrigate"

    logging.info(
    f"State: {state} | Action: {action} | Decision: {decision}"
)

    return {

        "state": state,
        "action": action,
        "decision": decision
    }