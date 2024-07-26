import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# 定义文件路径
movies_file = 'movies.dat'

# 加载电影元数据，指定编码格式为ISO-8859-1
movies_df = pd.read_csv(movies_file, sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'],
                        encoding='ISO-8859-1')

# 显示前几行数据
# print(movies_df.head())

# 创建TF-IDF矩阵和相似度矩阵
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['Genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 加载评分数据集
reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file('ratings.dat', reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)

# 提取训练集中的用户ID
train_user_ids = [trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()]
# print(f"训练集中可用的用户ID：{train_user_ids[:10]}")

# 检查是否存在已保存的模型
try:
    with open('svd_model.pkl', 'rb') as f:
        algo = pickle.load(f)
    print("模型已加载")
except FileNotFoundError:
    print("未找到已保存的模型，重新训练模型")
    # 定义参数网格
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }

    # 进行网格搜索
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)

    # 使用最佳参数训练模型
    algo = gs.best_estimator['rmse']
    algo.fit(trainset)

    # 保存模型到文件
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(algo, f)
    print("模型已保存")

# 评估模型
predictions = algo.test(testset)
accuracy.rmse(predictions)


# 基于协同过滤的推荐
def get_top_n_recommendations(algo, user_id, n=10):
    # 检查用户是否在训练集中
    try:
        inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    user_ratings = trainset.ur[inner_id]
    rated_movie_ids = [trainset.to_raw_iid(inner_id) for (inner_id, _) in user_ratings]
    movie_ids = [i for i in range(1, 3706) if i not in rated_movie_ids]
    predictions = [algo.predict(user_id, movie_id) for movie_id in movie_ids]
    # print(predictions)
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return recommendations


# 基于内容的推荐
def get_content_based_recommendations(movie_id, n=10):
    idx = movies_df.index[movies_df['MovieID'] == movie_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['MovieID'].iloc[movie_indices].tolist()


# 新用户冷启动推荐
def get_recommendations_for_new_user(n=10):
    popular_movies = movies_df['MovieID'].value_counts().index[:n]
    return popular_movies.tolist()


# 新电影冷启动推荐
def get_recommendations_for_new_movie(new_movie_genres, n=10):
    new_movie_tfidf = tfidf.transform([new_movie_genres])
    sim_scores = linear_kernel(new_movie_tfidf, tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['MovieID'].iloc[movie_indices].tolist()


# 测试

movie_id = 1
new_movie_genres = "Action|Adventure"

collaborative_recommendations = get_top_n_recommendations(algo, user_id=train_user_ids[0], n=10)
content_recommendations = get_content_based_recommendations(movie_id=movie_id, n=10)
new_user_recommendations = get_recommendations_for_new_user(n=10)
new_movie_recommendations = get_recommendations_for_new_movie(new_movie_genres, n=10)

print("协同过滤推荐:\n", collaborative_recommendations)
print(f"基于内容的推荐:(movieId=1)\n{content_recommendations}(值为推荐的moviesId)")
print(f"新用户推荐:\n{new_user_recommendations}(值为推荐的moviesId)")
print(f"新电影推荐:\n{new_movie_recommendations}(值为推荐的moviesId)")
