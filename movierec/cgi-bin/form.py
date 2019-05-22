#!/usr/bin/env python3
import cgi
import numpy as np
import pandas as pd


def fnctn(text1, text2):
    text1 = int(text1)
    text2 = int(text2)
    ratings_list = [i.strip().split(",") for i in
                    open('/home/anton/PycharmProjects/linear_course_proj/ml-latest-small/ratings_small.csv',
                         'r').readlines()]
    movies_list = [i.strip().split(",") for i in
                   open('/home/anton/PycharmProjects/linear_course_proj/ml-latest-small/movies.csv', 'r').readlines()]

    lst = [i for i in range(len(movies_list)) if len(movies_list[i]) != 3]
    lst = list(reversed(lst))
    for i in lst:
        del movies_list[i]

    ratings_df = pd.DataFrame(ratings_list[1:4648], columns=['userId', 'movieId', 'rating', 'timestamp'], dtype=float)

    movie_data = pd.DataFrame(movies_list[1:], columns=['movieId', 'title', 'genres'])
    movie_data['movieId'] = movie_data['movieId'].apply(pd.to_numeric)

    ratings_mat = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
    cov_mat = np.cov(normalised_mat)

    evals, evecs = np.linalg.eig(cov_mat)

    def print_smlr_movies(movie_data, text1, top_indexes):
        # print(movie_data[movie_data.movieId == movie_id].title.values[0])
        l = ""
        for id in top_indexes + 1:  # type: object
            while len(movie_data[movie_data.movieId == id].title.values) == 0:
                id = id + 1
            l+=str((movie_data[movie_data.movieId == id].title.values[0])+'<br>')
        return l
    def cos_similarity(data, movie_id, top_n=10):
        ind = movie_id - 1
        movie_row = data[ind, :]
        magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
        similarity = np.dot(movie_row, data.T) / (magnitude[ind] * magnitude)
        sort_indexes = np.argsort(-similarity)
        return sort_indexes[:top_n]

    k = 5
    # movie_id = 1
    # top_n = 8
    # print(evals)
    sliced = evecs[:, :k]
    top_ind = cos_similarity(sliced, text1, text2)
    # print(top_ind)
    text = print_smlr_movies(movie_data, text1, top_ind)
    return text


form = cgi.FieldStorage()
text1 = form.getfirst("TEXT_1", 1)
text2 = form.getfirst("TEXT_2", 10)
print(text1, text2)
print("Content-type: text/html\n")
print("""<!DOCTYPE HTML>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Обработка данных форм</title>
        </head>
        <body style="background-image: url( ../theatre.jpg ) !important; height: 100%;vertical-align: center">""")
print("<center><h1 style = 'color: white; border-bottom: 5px solid #f40024; margin-top:10%'>Результат:</h1></center>")
l = fnctn(text1, text2)
print("<center><p style = 'color:white; display:flex; flex-direction: column; font-size:36px; justify-content: center'>{}</p>".format(l))

print("""</body>
        </html>""")
