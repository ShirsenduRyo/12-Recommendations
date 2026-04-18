import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


class DatasetLoader:

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load(self, dataset_name):

        if dataset_name == "open_bandit":
            return self.load_open_bandit()

        elif dataset_name == "movielens":
            return self.load_movielens()

        elif dataset_name == "mimic":
            return self.load_mimic()

        else:
            raise ValueError("Unknown dataset")

    # -----------------------------------------------------
    # 1. OPEN BANDIT DATASET
    # -----------------------------------------------------

    def load_open_bandit(self):

        path = os.path.join(self.data_dir, "open_bandit", "obd.csv")

        df = pd.read_csv(path)

        context = df.filter(regex="context").values
        action = df["action"].values
        reward = df["reward"].values
        logging_prob = df["propensity"].values

        target_prob = np.ones_like(logging_prob)

        timestamp = np.arange(len(df))

        scaler = StandardScaler()
        context = scaler.fit_transform(context)

        return dict(
            context=context,
            action=action,
            reward=reward,
            logging_prob=logging_prob,
            target_prob=target_prob,
            timestamp=timestamp
        )

    # -----------------------------------------------------
    # 2. MOVIELENS
    # -----------------------------------------------------

    def load_movielens(self):

        ratings_path = os.path.join(
            self.data_dir,
            "movielens",
            "ratings.csv"
        )

        movies_path = os.path.join(
            self.data_dir,
            "movielens",
            "movies.csv"
        )

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

        df = ratings.merge(movies, on="movieId")

        df["reward"] = (df["rating"] >= 4).astype(int)

        df["action"] = df["movieId"].astype("category").cat.codes

        context = df[["userId"]].values

        logging_prob = np.ones(len(df)) * 0.2

        target_prob = np.ones(len(df)) * 0.2

        timestamp = df["timestamp"].values

        context = StandardScaler().fit_transform(context)

        next_context = np.roll(context, -1, axis=0)

        return dict(
            context=context,
            action=df["action"].values,
            reward=df["reward"].values,
            logging_prob=logging_prob,
            target_prob=target_prob,
            timestamp=timestamp,
            next_context=next_context
        )

    # -----------------------------------------------------
    # 3. MIMIC
    # -----------------------------------------------------

    def load_mimic(self):

        path = os.path.join(
            self.data_dir,
            "mimic",
            "mimic_processed.csv"
        )

        df = pd.read_csv(path)

        context = df.filter(regex="feature").values

        action = df["treatment"].values

        reward = df["outcome"].values

        logging_prob = np.ones(len(df)) * 0.25

        target_prob = np.ones(len(df)) * 0.25

        timestamp = df["time"].values

        next_context = np.roll(context, -1, axis=0)

        context = StandardScaler().fit_transform(context)

        return dict(
            context=context,
            action=action,
            reward=reward,
            logging_prob=logging_prob,
            target_prob=target_prob,
            timestamp=timestamp,
            next_context=next_context
        )