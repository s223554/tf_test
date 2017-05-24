import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.random as rnd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split


tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

# load data
MOVIE_PATH = "dataset"
def load_data(path=MOVIE_PATH):
    csv_path = os.path.join(path, "movie_metadata.csv")
    return pd.read_csv(csv_path)

raw_data = load_data()
attri_train = ["duration", "cast_total_facebook_likes", "num_user_for_reviews"]

train_set,test_set = train_test_split(raw_data,test_size=0.2,random_state=10)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


num_pipeline = Pipeline([
#    ('selector',DataFrameMapper(attributes))
    ('selector',DataFrameSelector(attri_train)),
    ('imputer',Imputer(strategy='median')),
#    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

data_train = num_pipeline.fit_transform(train_set)
train_y = train_set['imdb_score']
m = len(data_train)
n = int(data_train.size/m)

train_data_plus_bias = np.c_[np.ones((m, 1)), data_train]

X = tf.constant(train_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(train_y, dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = train_data_plus_bias[indices]
    y_batch = train_y[indices]
    return X_batch, y_batch

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)