import pickle
import re
import nltk
import pandas as pd
import string
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    last_three_columns = df.columns[-3:]
    rows_to_drop = [209538, 220731, 221678]

    books = df.drop(columns=last_three_columns)
    books = books.drop(index=rows_to_drop)

    new_df = books.dropna()
    new_df['Year-Of-Publication'] = pd.to_numeric(new_df['Year-Of-Publication'], errors='coerce')
    df_filtered = new_df.loc[new_df["Year-Of-Publication"] <= 2024]
    df_filtered.info()

    return df_filtered


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df[df['Book-Rating'] != 0.0]

    books_with_one_review = ratings['ISBN'].value_counts()[ratings['ISBN'].value_counts() == 1].index
    users_with_one_review = ratings['User-ID'].value_counts()[ratings['User-ID'].value_counts() == 1].index

    filtered_rating_df = ratings[~ratings['ISBN'].isin(books_with_one_review) & ~ratings['User-ID'].isin(users_with_one_review)]
    
    return filtered_rating_df


def title_preprocessing(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    new_book = ratings.groupby("ISBN")["Book-Rating"].mean().reset_index()
    new_book.rename(columns={"Book-Rating": "Average-Rating"}, inplace=True)

    data = books.merge(new_book, on="ISBN", how="inner")
    data.dropna(subset=["Average-Rating"], inplace=True)

    X = data[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
    Y = data["Average-Rating"]

    preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(), 'Book-Title'),
        ('num', StandardScaler(), ['Year-Of-Publication'])
    ],
    remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(max_iter=1000, tol=1e-3))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(pipeline, file)


books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
new_books = books_preprocessing(books)
new_ratings = ratings_preprocessing(ratings)
nltk.download('punkt_tab')
nltk.download('stopwords')
new_books["Book-Title"] = books["Book-Title"].apply(title_preprocessing)

modeling(new_books, new_ratings)
