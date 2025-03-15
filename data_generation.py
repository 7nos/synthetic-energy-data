# scripts/data_generation.py
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)

def generate_data(n_households=10_000):
    data = {
        "household_id": [fake.uuid4() for _ in range(n_households)],
        "city": [fake.city() for _ in range(n_households)],
        "occupants": np.random.randint(1, 6, n_households),
        "income_bracket": np.random.choice(
            ["Low", "Medium", "High"], 
            size=n_households, 
            p=[0.4, 0.4, 0.2]
        ),
        "home_size_sqft": np.random.randint(800, 3000, n_households),
    }

    # Solar panel logic
    data["has_solar_panel"] = np.where(
        pd.Series(data["income_bracket"]) == "High",
        np.random.choice([0, 1], size=n_households, p=[0.2, 0.8]),
        np.random.choice([0, 1], size=n_households, p=[0.7, 0.3])
    )

    # Energy consumption with noise
    data["daily_kwh"] = np.where(
        pd.Series(data["income_bracket"]) == "High",
        np.random.uniform(15, 30, n_households),
        np.random.uniform(20, 50, n_households)
    )
    data["daily_kwh"] = np.round(data["daily_kwh"] * np.random.normal(1, 0.02, n_households), 2)

    # Missing values in thermostat
    data["thermostat_setting"] = np.random.randint(60, 80, n_households).astype(float)
    mask = np.random.choice([True, False], size=n_households, p=[0.05, 0.95])
    data["thermostat_setting"] = np.where(mask, np.nan, data["thermostat_setting"])

    # Temperature and seasonality
    data["avg_temp"] = np.round(np.random.uniform(15, 35, n_households), 1)
    data["month"] = np.random.randint(1, 13, n_households)
    summer_mask = pd.Series(data["month"]).isin([6, 7, 8])
    data["daily_kwh"] = np.where(summer_mask, data["daily_kwh"] * 1.2, data["daily_kwh"])

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("D:\kaggle_dataset\synthetic_energy_data.csv", index=False)