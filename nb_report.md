# Part 2 - Naive Bayes with MapReduce

## Dataset
AG News - news articles (title + description), 4 classes:
1 World, 2 Sports, 3 Business, 4 Sci/Tech.
120,000 articles, 30,000 per class.

Hosted at: https://raw.githubusercontent.com/jawahar1609/cs6350-data/main/train.csv.gz

## Preprocessing
- Title + description joined, lowercased
- Tokenize: keep only `[a-z]+`
- Drop stop words (Spark's English list) and words of 2 letters or fewer
- Porter stemming (nltk)
- 80/20 random split, seed 42

## Model
Multinomial Naive Bayes with Laplace smoothing:

```
P(c)     = docs in c / total docs
P(w | c) = (count(w, c) + 1) / (total words in c + |V|)
predict  = argmax_c  log P(c) + sum over words of log P(w | c)
```

Words not seen in training are skipped.

## Pseudo-code

```
# priors
map(label, words)          -> (label, 1)
reduceByKey(sum)           -> (label, docs_c)
prior[c] = docs_c / N

# word counts
map(label, words)          -> for w in words: ((label, w), 1)
reduceByKey(sum)           -> ((label, w), count)

# class totals + vocab
map((label, w), count)     -> (label, count)
reduceByKey(sum)           -> total[c]
V = distinct words

# likelihoods
map((label, w), count)     -> (w, (label, log((count + 1) / (total[label] + V))))
groupByKey                 -> w -> {c: log P(w|c)}
broadcast model

# test
map(label, words)          -> (label, argmax_c log prior[c] + sum log P(w|c))
accuracy = correct / total
```

## Results

| | |
|---|---|
| Train docs | 95,987 |
| Test docs | 24,013 |
| Vocab size | 39,287 |
| **Accuracy** | **90.31%** |

**Class priors**

| Class | Prior |
|---|---|
| World | 0.2499 |
| Sports | 0.2496 |
| Business | 0.2507 |
| Sci/Tech | 0.2498 |

**Confusion matrix** (rows = actual, cols = predicted)

| | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|
| World | 5382 | 193 | 262 | 172 |
| Sports | 50 | 5932 | 40 | 24 |
| Business | 224 | 61 | 5079 | 574 |
| Sci/Tech | 232 | 58 | 436 | 5294 |

**Top words per class**
- World: said, kill, iraq, reuter, presid, minist, two, say, new, afp
- Sports: game, win, first, new, team, year, season, two, one, night
- Business: reuter, said, new, oil, stock, compani, price, year, inc, profit
- Sci/Tech: new, microsoft, compani, said, softwar, servic, reuter, internet, year, comput

## Observations
- Priors are almost equal because the dataset is balanced.
- Sports is the easiest class (98% recall). Its words (game, team, season) rarely show up anywhere else.
- Most errors are Business <-> Sci/Tech (574 + 436). Tech company news (Microsoft, Google, earnings) uses words from both classes.
- 90.3% is in line with what's usually reported for Naive Bayes on AG News.
