from fastapi import FastAPI
from pydantic import BaseModel

import logging
import pickle
import os

from dotenv import load_dotenv

from sim.q_learning_agent import QLearningAgent


# ======================================
# LOAD ENV VARIABLES
# ======================================

load_dotenv()


# ======================================
# LOGGING CONFIGURATION
# ======================================

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# ======================================
# LOAD TRAINED MODEL
# ======================================

model_path = os.getenv("MODEL_PATH")

agent = QLearningAgent()

with open(model_path, "rb") as file:

    agent.q_table = pickle.load(file)


# ======================================
# CREATE FASTAPI APP
# ======================================

app = FastAPI(
    title="Smart Irrigation API",
    description="Reinforcement Learning based Smart Irrigation Optimization System",
    version="1.0"
)


# ======================================
# INPUT SCHEMA
# ======================================

class IrrigationInput(BaseModel):

    soil_moisture: int
    temperature: int
    rainfall: int


# ======================================
# ROOT ROUTE
# ======================================

@app.get("/")

def home():

    return {
        "message": "Smart Irrigation API Running Successfully"
    }


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

    # ======================================
    # LOG PREDICTIONS
    # ======================================

    logging.info(
        f"State: {state} | "
        f"Action: {action} | "
        f"Decision: {decision}"
    )

    return {

        "state": state,
        "action": action,
        "decision": decision
    }