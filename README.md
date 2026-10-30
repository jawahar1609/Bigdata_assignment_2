# Assignment 2 - Part 1

## Needs
- Python 3, Java 11/17, PySpark (`pip install pyspark`)
- Internet (data is read from the S3 URL)

## Part 1.1 - MapReduce
```
spark-submit friend_recommendation_mr.py
```
Prints 10 random users and their top 10 recs (`UserID<TAB>rec1,...,rec10`).
Saves `sample_users.txt` and `mapreduce_recs.txt`.

## Part 1.2 - ALS
Run after 1.1, from the same folder (so it uses the same 10 users):
```
spark-submit --driver-memory 4g friend_recommendation_als.py
```
Prints the best hyperparameters, the test RMSE, the top 10 recs for the same 10 users, and the overlap with the MapReduce recs.
Saves `als_recs.txt`.

If `sample_users.txt` is missing, it picks the same 10 users using seed 42.

## Part 2 - Naive Bayes
Also needs nltk (`pip install nltk`).
```
spark-submit naive_bayes_mr.py
```
Reads the AG News CSV from its public URL. Prints the train/test sizes, vocab size, class priors, accuracy, confusion matrix and top words per class.
