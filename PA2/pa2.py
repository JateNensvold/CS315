import pandas as pd
import pickle


def merge() -> pd.DataFrame:
    r'''
    Merge ratings and movies together into a single matrix based on movieID

    Args:
        None
    Output:
        Pandas Dataframe containing movie info
    '''
    ratings = pd.read_csv("./movie-data/ratings.csv")
    movies = pd.read_csv("./movie-data/movies.csv")

    df = pd.merge(ratings, movies, on="movieId")
    print("Merged")

    return df


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    r'''
    Build a pivot table with userId, titles and ratings from a pandas dataframe/matrix

    Args:
        df: a pandas dataframe containg movie data
    Returns:
        Pandas Dataframe containing formated movie data
    '''
    movie_matrix = df.pivot_table(
        index="userId", columns="title", values='rating')
    print("Pivot")
    return movie_matrix


def correlate(movie_matrix: pd.DataFrame) -> pd.DataFrame:
    r'''
    Create a correlation matrix between different movies

    Args:
        movie_matrix: a formated movie matrix

    Return:
        Pandas Dataframe containing movie to movie correlation matrix
    '''
    corr_matrix = movie_matrix.corr(method="pearson", min_periods=50)

    # Create a pickle object of the corr_matrix
    with open("correlation.pickle", 'wb') as handle:
        pickle.dump(corr_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Correlation")
    return corr_matrix


if __name__ == "__main__":

    # build matrix
    df = merge()
    movie_matrix = pivot(df)
    # correlate the formated matrix with movie to movie filter
    corr_matrix = correlate(movie_matrix)

    # read in movies.csv to do a lokoup between movie name/id
    movies = pd.read_csv("./movie-data/movies.csv")
    # open output.txt for results
    with open("output.txt", 'w') as f:
        for i in range(1, len(movie_matrix)):  # Len(movie_matrix)
            userratings = movie_matrix.iloc[i].dropna()
            # print("Ratings for user ", str(i))
            # print(userratings)
            recommend = pd.Series(dtype='float64')

            for j in range(0, len(userratings)):
                # add movies that are similar to users i
                # Find how similar movies are
                similar = corr_matrix[userratings.index[j]].dropna()
                similar = similar.map(lambda x: x * userratings[j])
                recommend = recommend.append(similar)

            # print("Sorting Recomendations")
            recommend.sort_values(inplace=True, ascending=False)
            print(i, end=" ", file=f)
            temp = recommend.head(n=5)

            x = pd.DataFrame(recommend)
            recommend_filter = x[~x.index.isin(userratings.index)]
            recommend_filter: pd.DataFrame
            # get top 5 movie ratings and movie tiltes
            tempVal1 = recommend_filter.head(5)
            for k, v in tempVal1[0].items():
                # get movie id from title
                tempVal = movies.loc[movies['title'] == k]
                # output movie id to results file
                print(tempVal.index.tolist()[0], end=" ", file=f)
            print(file=f)
            # User-id1 movie-id1 movie-id2 movie-id3 movie-id4 movie-id5
