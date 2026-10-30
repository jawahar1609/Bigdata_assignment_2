import os
import sys
import random
from pyspark.sql import SparkSession, functions as F
from pyspark import SparkFiles
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

URL = "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"
SEED = 42

spark = (SparkSession.builder.appName("FriendRecsALS")
         .config("spark.driver.memory", "4g")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("ERROR")

src = sys.argv[1] if len(sys.argv) > 1 else URL
if src.startswith("http"):
    sc.addFile(src)
    src = "file://" + SparkFiles.get(os.path.basename(src))


def parse(line):
    p = line.strip().split("\t")
    fr = [int(x) for x in p[1].split(",") if x] if len(p) > 1 else []
    return int(p[0]), fr


data = sc.textFile(src).filter(lambda l: l.strip()).map(parse).cache()

if os.path.exists("sample_users.txt"):
    with open("sample_users.txt") as f:
        users = [int(x) for x in f.read().strip().split(",")]
else:
    users = sorted(data.keys().takeSample(False, 10, seed=SEED))

pos = data.flatMap(lambda r: [(r[0], x) for x in r[1]]).cache()

ids = sc.broadcast(data.keys().collect())


# ALS needs 0s too, otherwise every rating is 1 and RMSE means nothing
def neg_sample(idx, it):
    rnd = random.Random(SEED + idx)
    for u, _ in it:
        yield u, rnd.choice(ids.value)


neg = (pos.mapPartitionsWithIndex(neg_sample)
       .filter(lambda p: p[0] != p[1])
       .distinct()
       .subtract(pos))

ratings = spark.createDataFrame(
    pos.map(lambda p: (p[0], p[1], 1.0)).union(neg.map(lambda p: (p[0], p[1], 0.0))),
    ["user", "item", "rating"]).cache()

u_idx = StringIndexer(inputCol="user", outputCol="u", handleInvalid="skip")
i_idx = StringIndexer(inputCol="item", outputCol="i", handleInvalid="skip")
als = ALS(userCol="u", itemCol="i", ratingCol="rating",
          coldStartStrategy="drop", seed=SEED)

pipe = Pipeline(stages=[u_idx, i_idx, als])

grid = (ParamGridBuilder()
        .addGrid(als.rank, [10, 20])
        .addGrid(als.regParam, [0.01, 0.1])
        .addGrid(als.maxIter, [5, 10])
        .build())

ev = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=ev,
                    numFolds=3, seed=SEED, parallelism=2)

train, test = ratings.randomSplit([0.8, 0.2], seed=SEED)
model = cv.fit(train).bestModel

best = model.stages[-1]
print(f"best rank={best.rank} regParam={best._java_obj.parent().getRegParam()} "
      f"maxIter={best._java_obj.parent().getMaxIter()}")
print(f"test RMSE = {ev.evaluate(model.transform(test)):.4f}")

u_map = spark.createDataFrame([(i, int(x)) for i, x in enumerate(model.stages[0].labelsArray[0])], ["u", "user"])
i_map = spark.createDataFrame([(i, int(x)) for i, x in enumerate(model.stages[1].labelsArray[0])], ["i", "item"])

uf = (best.userFactors.withColumnRenamed("features", "uf")
      .join(u_map, best.userFactors.id == u_map.u)
      .filter(F.col("user").isin(users))
      .select("user", "uf"))
itf = (best.itemFactors.withColumnRenamed("features", "itf")
       .join(i_map, best.itemFactors.id == i_map.i)
       .select("item", "itf"))

friends = spark.createDataFrame(pos.filter(lambda p: p[0] in set(users)), ["user", "item"])

scores = (uf.crossJoin(itf)
          .filter(F.col("user") != F.col("item"))
          .join(friends, ["user", "item"], "left_anti")
          .withColumn("score", F.expr("aggregate(zip_with(uf, itf, (a, b) -> a * b), 0D, (s, x) -> s + x)"))
          .select("user", "item", "score")
          .collect())

recs = {}
for r in scores:
    recs.setdefault(r.user, []).append((r.item, r.score))

res = []
for u in users:
    top = [x for x, _ in sorted(recs.get(u, []), key=lambda t: (-t[1], t[0]))[:10]]
    res.append(f"{u}\t{','.join(map(str, top))}")
print("\n".join(res))

with open("als_recs.txt", "w") as f:
    f.write("\n".join(res) + "\n")

if os.path.exists("mapreduce_recs.txt"):
    with open("mapreduce_recs.txt") as f:
        mr = {l.split("\t")[0]: set(l.strip().split("\t")[1].split(",")) - {""}
              for l in f if "\t" in l}
    print("\noverlap with MapReduce (out of 10)")
    for line in res:
        u, r = line.split("\t")
        print(f"{u}\t{len(mr.get(u, set()) & set(r.split(',')) - {''})}")

spark.stop()
