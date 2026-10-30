import os
import re
import sys
from math import log
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml.feature import StopWordsRemover

URL = "https://raw.githubusercontent.com/jawahar1609/cs6350-data/main/train.csv.gz"
SEED = 42
NAMES = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}

spark = SparkSession.builder.appName("NaiveBayesMR").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

src = sys.argv[1] if len(sys.argv) > 1 else URL
if src.startswith("http"):
    sc.addFile(src)
    src = "file://" + SparkFiles.get(os.path.basename(src))

raw = spark.read.csv(src, quote='"', escape='"').rdd

stop = sc.broadcast(set(StopWordsRemover.loadDefaultStopWords("english")))


def clean(rows):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    for r in rows:
        txt = f"{r[1]} {r[2]}".lower().replace("\\", " ")
        toks = [ps.stem(w) for w in re.findall(r"[a-z]+", txt)
                if len(w) > 2 and w not in stop.value]
        if toks:
            yield r[0], toks


docs = raw.mapPartitions(clean).cache()
train, test = docs.randomSplit([0.8, 0.2], seed=SEED)
train.cache()

n = train.count()
cnt = train.map(lambda d: (d[0], 1)).reduceByKey(add).collectAsMap()
prior = {c: k / n for c, k in cnt.items()}
classes = sorted(prior)

wc = (train.flatMap(lambda d: [((d[0], w), 1) for w in d[1]])
      .reduceByKey(add)
      .cache())

tot = wc.map(lambda kv: (kv[0][0], kv[1])).reduceByKey(add).collectAsMap()
V = wc.map(lambda kv: kv[0][1]).distinct().count()

model = (wc.map(lambda kv: (kv[0][1], (kv[0][0], log((kv[1] + 1) / (tot[kv[0][0]] + V)))))
         .groupByKey()
         .mapValues(dict)
         .collectAsMap())

bm = sc.broadcast(model)
unk = {c: log(1 / (tot[c] + V)) for c in classes}
lp = {c: log(p) for c, p in prior.items()}


def predict(words):
    m = bm.value
    best, best_s = None, None
    for c in classes:
        s = lp[c] + sum(m[w].get(c, unk[c]) for w in words if w in m)
        if best_s is None or s > best_s:
            best, best_s = c, s
    return best


preds = test.map(lambda d: (d[0], predict(d[1]))).cache()
m_test = preds.count()
acc = preds.filter(lambda p: p[0] == p[1]).count() / m_test
cm = preds.countByValue()

print(f"\ntrain docs: {n}   test docs: {m_test}   vocab: {V}")
print("\nclass priors")
for c in classes:
    print(f"  {c} {NAMES.get(c, c):<9} {prior[c]:.4f}")

print(f"\naccuracy: {acc:.4f}")

print("\nconfusion matrix (rows = actual, cols = predicted)")
print("          " + "".join(f"{NAMES.get(c, c):>10}" for c in classes))
for a in classes:
    print(f"{NAMES.get(a, a):<10}" + "".join(f"{cm.get((a, p), 0):>10}" for p in classes))

print("\ntop words per class")
top = (wc.map(lambda kv: (kv[0][0], (kv[1], kv[0][1])))
       .groupByKey()
       .mapValues(lambda v: [w for _, w in sorted(v, reverse=True)[:10]])
       .collectAsMap())
for c in classes:
    print(f"  {NAMES.get(c, c):<9} {', '.join(top[c])}")

spark.stop()
