# Donor Segmentation: KDD Cup 1998

Customer segmentation of 95,412 lapsed donors to a US veterans' charity (the KDD Cup 1998 PVA learning set), to tailor direct-mail campaigns by segment.

The full analysis, with outputs and figures, is in [`donor_segmentation.ipynb`](donor_segmentation.ipynb).

## Approach

1. **Feature engineering:** donor age and years since first gift at the June 1997 campaign (dates in the raw file are `YYMM` integers), urbanicity from the `DOMAIN` code, log transforms of skewed amounts, and a set of donor-interest flags.
2. **Preparation:** KNN imputation of income and age, an inter-quantile outlier filter (97% of donors kept), standardisation, one-hot encoding of categoricals.
3. **Clustering tendency:** VAT and the Hopkins statistic (0.12: clearly clusterable).
4. **Clustering:** K-means (k chosen by elbow and Calinski-Harabasz), Ward hierarchical clustering on top of 30 K-means centroids, and clustering by perspective (value features vs engagement features) merged with Ward linkage.
5. **Profiling:** cluster means on the original scale, parallel-coordinate plots for metric, state, gender, urbanicity and interest features, and t-SNE views on a 5,000-donor sample.

## Segments (K-means, k = 4)

| Segment | Profile |
|---|---|
| Loyal veterans | ~9.6 years of giving, ~20 gifts, highest lifetime total, small average gift, most card-mailing responses |
| New generous donors | under 2 years, ~2 gifts, largest minimum and average gifts |
| Lapsing mid-tenure donors | ~6 years, ~7 gifts, largest last gift, longest first-to-second gift gap |
| Frequent small givers | highest gift frequency, smallest recent gifts |

Giving behaviour separates donors much more than demographics do.

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute donor_segmentation.ipynb --output executed.ipynb
```

The pinned stack is the 2021-era one that `pyclustertend` requires (scikit-learn 0.24, pandas 1.3, Python 3.9). The data (36 MB zip) is downloaded from the [UCI KDD Archive](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html) on first run. The GitHub Actions workflow executes the notebook on every push.
