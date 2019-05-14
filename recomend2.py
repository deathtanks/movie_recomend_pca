import numpy as np
import pandas as pd

ratings_list = [i.strip().split(",") for i in open('/home/anton/PycharmProjects/linear_course_proj/ml-latest-small/ratings.csv', 'r').readlines()]
movies_list = [i.strip().split(",") for i in open('/home/anton/PycharmProjects/linear_course_proj/ml-latest-small/movies.csv', 'r').readlines()]

lst = [i for i in range(len(movies_list)) if len(movies_list[i]) != 3]
lst = list(reversed(lst))
for i in lst:
   del movies_list[i]

ratings_df = pd.DataFrame(ratings_list[1:], columns = ['userId','movieId','rating','timestamp'], dtype = float)
movie_data = pd.DataFrame(movies_list[1:], columns = ['movieId', 'title', 'genres'])
movie_data['movieId'] = movie_data['movieId'].apply(pd.to_numeric)

ratings_mat = ratings_df.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)
normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
cov_mat = np.cov(normalised_mat)
for i in cov_mat:
    for j in i:
        round(j, 3)
evals, evecs = np.linalg.eig(cov_mat)

def print_smlr_movies(movie_data, movie_id, top_indexes):
    print(movie_data[movie_data.movie_id == movie_id].title.values[0])
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

def cos_similarity(data, movie_id, top_n=10):
    ind = movie_id - 1
    movie_row = data[ind, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[ind] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

k = 3
movie_id = 1
top_n = 3

sliced = evecs[:, :k]
top_ind = cos_similarity(sliced, movie_id, top_n)
print_smlr_movies(movie_data, movie_id, top_ind)
