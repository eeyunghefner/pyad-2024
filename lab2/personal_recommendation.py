def personal_recommedations(books, ratings, lin_model, svd_model):
  ratings_zero = ratings[ratings["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
  current_user = ratings_zero.idxmax()

  zero_books = ratings[(ratings["User-ID"] == current_user) & (ratings["Book-Rating"] == 0)]["ISBN"].unique()

  test_svd = []
  for elem in zero_books:
    test_svd.append((current_user, elem, 0))
  predictions_svd = svd_model.test(test_svd)

  eight_books = []
  for element in predictions_svd:
    if element.est >= 8:
      eight_books.append(element.iid)
  
  books_data = books[books["ISBN"].isin(eight_books)].copy()

  X_test = books_data[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
  prediction_lin = lin_model.predict(X_test)

  books_data["Predicted-Rating"] = prediction_lin
  books_data.sort_values("Predicted-Rating", ascending=False, inplace=True)

  recommendation = books_data[["ISBN", "Book-Title", "Book-Author", "Predicted-Rating"]].head(10)

  print("10 книг по рейтингу")
  print(recommendation.to_string(index=False))

'''
10 книг по рейтингу
      ISBN                                                            Book-Title          Book-Author  Predicted-Rating
0451194861                             Wizard and Glass (The Dark Tower, Book 4)         Stephen King          8.287668
059035342X      Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))        J. K. Rowling          8.205646
0439064872                      Harry Potter and the Chamber of Secrets (Book 2)        J. K. Rowling          8.152329
0694003611                                             Goodnight Moon Board Book  Margaret Wise Brown          8.055836
0064409422 The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition)          C. S. Lewis          8.006965
0064400018                                         Little House in the Big Woods Laura Ingalls Wilder          7.966624
0451160525                               The Gunslinger (The Dark Tower, Book 1)         Stephen King          7.966348
0446364193                               Along Came a Spider (Alex Cross Novels)      James Patterson          7.915751
0064400050                           By the Shores of Silver Lake (Little House) Laura Ingalls Wilder          7.908066
0440406498                The Black Cauldron (Chronicles of Prydain (Paperback))      LLOYD ALEXANDER          7.899793
'''
