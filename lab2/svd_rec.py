import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df[df['Book-Rating'] != 0.0]

    books_with_one_review = ratings['ISBN'].value_counts()[ratings['ISBN'].value_counts() == 1].index
    users_with_one_review = ratings['User-ID'].value_counts()[ratings['User-ID'].value_counts() == 1].index

    filtered_rating_df = ratings[~ratings['ISBN'].isin(books_with_one_review) & ~ratings['User-ID'].isin(users_with_one_review)]

    return filtered_rating_df


def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    ratings_data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)

    train_data, test_data = train_test_split(ratings_data, test_size=0.2)

    model = SVD()
    model.fit(train_data)

    predictions = model.test(test_data)
    mae = accuracy.mae(predictions)
    print(f"MAE on test set: {mae}")

    with open("svd.pkl", "wb") as file:
        pickle.dump(model, file)


ratings = pd.read_csv("Ratings.csv")
new_ratings = ratings_preprocessing(ratings)
modeling(new_ratings)
