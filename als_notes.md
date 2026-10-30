# Part 1.2 - ALS Friend Recommendation

## Preprocessing
- Read the adjacency list from the S3 URL and split each line into (user, friends).
- Every friendship becomes a rating `(user, friend, 1.0)`. The graph is undirected, so both directions are already in the file.
- With only 1s, ALS can just predict 1 for everything and the RMSE means nothing. So for every positive pair we pick a random user who isn't a friend and add `(user, random, 0.0)`. That gives about a 50/50 split of 1s and 0s.
- The same user ID is used as both the "user" and the "item".

## Model
- Pipeline: `StringIndexer(user) -> StringIndexer(item) -> ALS`
- ALS is explicit, with `coldStartStrategy="drop"` so unseen IDs don't produce NaN in the RMSE.
- Grid (8 combos):
  - rank: 10, 20
  - regParam: 0.01, 0.1
  - maxIter: 5, 10
- CrossValidator, 3 folds, RMSE. Fit on 80% of the data; the RMSE is reported on the other 20%.

## Recommendations
- Score = dot product of the user's factor vector and each item's factor vector.
- Remove the user themselves and their existing friends, sort by score, keep the top 10.
- Same 10 users as Part 1.1 (read from `sample_users.txt`).

## Results and comparison
The script prints the best rank / regParam / maxIter and the test RMSE when it runs.
The ALS vs MapReduce comparison is in `comparison.md`.
