import os
import sys
from pyspark.sql import SparkSession
from pyspark import SparkFiles

URL = "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"
SEED = 42

spark = SparkSession.builder.appName("FriendRecs").getOrCreate()
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

users = sorted(data.keys().takeSample(False, 10, seed=SEED))
with open("sample_users.txt", "w") as f:
    f.write(",".join(map(str, users)))

picked = sc.broadcast(set(users))


def mapper(rec):
    u, fr = rec
    out = []
    for a in fr:
        if a not in picked.value:
            continue
        out.append(((a, u), (0, 1)))
        for b in fr:
            if a != b:
                out.append(((a, b), (1, 0)))
    return out


recs = (data.flatMap(mapper)
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        .filter(lambda kv: kv[1][1] == 0)
        .map(lambda kv: (kv[0][0], (kv[0][1], kv[1][0])))
        .groupByKey()
        .mapValues(lambda v: [b for b, _ in sorted(v, key=lambda t: (-t[1], t[0]))[:10]])
        .collectAsMap())

res = [f"{u}\t{','.join(map(str, recs.get(u, [])))}" for u in users]
print("\n".join(res))

with open("mapreduce_recs.txt", "w") as f:
    f.write("\n".join(res) + "\n")

spark.stop()
