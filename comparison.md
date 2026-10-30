# ALS vs MapReduce - Friend Recommendations

Same 10 users (seed 42), top 10 recommendations each.

## Results

| User | MapReduce (mutual friends) | ALS | Overlap |
|---|---|---|---|
| 1760 | 1777,1745,1751,1755,1767,1742,1747,1756,1774,1786 | 1745,1545,1586,1534,1751,1550,1755,1777,1767,1590 | 5 |
| 3068 | 3052,3054,3048,3053,3081,3092,3103,3041,3062,3075 | 3053,3093,3089,3062,3044,3048,3060,3073,3092,3055 | 4 |
| 6456 | 6466,6508,6459,6542,6551,6481,6591,6530,6600,39157 | 6466,6508,6542,6671,41335,10782,6565,6487,6459,1356 | 4 |
| 6685 | 6483,6497,6542,6450,6456,6482,6495,6498,6536,6559 | 6671,41335,40664,28820,6542,12272,6579,38202,10792,1520 | 1 |
| 9117 | 9114,9130,9108,9111,9121,9127,9131,9109,9112,9118 | 9130,548,18810,2869,9114,12703,17343,35805,634,18845 | 2 |
| 29060 | 29048,29051,29056,29061,29045,29054,29064,29040,29041,29043 | 29056,29048,29040,29054,29051,29072,29041,29064,29044,29039 | 7 |
| 37604 | 36994,6541,10071,10102,18918,23273,24338,36824,36984,37091 | 37084,44132,37197,36933,36808,37047,44065,36892,37002,36775 | 0 |
| 44405 | 30711,154,562,6493,6565,8147,9912,12649,14277,18096 | 27312,6802,4104,30870,19211,12210,22623,14277,7357,18119 | 1 |
| 45577 | 6450,6496,6497,19034,39157,6466,6481,6482,6483,6493 | 41753,1787,3394,1511,1680,40119,6469,6955,6961,32751 | 0 |
| 48346 | 48338,48355,48357,48390,48405,48411,48333,48384,48385,48387 | 48357,48387,106,48424,48390,48384,48427,48338,48406,48411 | 6 |

**Total overlap: 30 / 100 (avg 3 per user).**
Of MapReduce's top 3 picks, 15 / 30 also show up in ALS's top 10.

## Differences

**1. Tight communities -> both agree.**
29060 (7), 48346 (6) and 1760 (5) have friends packed in one ID range. Both methods recommend from the same group, and ALS mostly picks the users MapReduce ranks highest.

**2. MapReduce stays local, ALS can jump.**
For 9117, MapReduce picks only 91xx users (the ones sharing the most friends). ALS keeps 9114 and 9130 but fills the rest with far-off users (548, 18810, 35805). It's matching on similar connection patterns, not on shared friends.

**3. Mixed friend lists -> ALS picks the dominant group.**
37604's mutual friends come from several groups, so MapReduce mixes them (6541, 10071, 18918...). ALS only picks from 36xxx-37xxx and 44xxx, which is where most of 37604's friends are. Overlap 0.

**4. Scattered users -> they disagree most.**
44405 and 45577 have friends spread across groups. MapReduce still has exact counts to rank by. ALS's latent vectors for these users are weaker, so its picks look random (0-1 overlap).

**5. ALS repeats popular users.**
41335 and 6671 show up for both 6456 and 6685. Their vectors line up with many users, so ALS ranks them high even with few mutual friends. MapReduce never does this; it only counts real mutual friends.

## Summary

| | MapReduce | ALS |
|---|---|---|
| Based on | Exact mutual-friend count | Learned latent factors |
| Explainable | Yes ("12 friends in common") | No |
| Can recommend 0-mutual users | No | Yes |
| Best for | Dense local groups | Finding similar users beyond 2 hops |
| Cost | One MapReduce pass | Training + CV grid search |

MapReduce is the safer pick for "people you may know". ALS finds wider, less obvious matches, but some of them are noisy, especially for users with scattered or few friends.
