# Changelog

## Unreleased

## Added

* Added `datasets.load_wolf_or_huksy`, which a small dataset of images that either depict wolves with a snowy background, or huskies with a grassy background.

## 0.0.4 - 13/12/19

### Added

* `BaseExplainer.plot_weights()` to visualize how many individuals capture X% of the weight used to stress the distribution.
* `CacheExplainer.plot_influence_2d()` and `CacheExplainer.plot_performance_2d()` to reach two means simultaneously (and so considering correlations between features).
* `link_variables` parameter to `CacheExplainer.explain_influence()` and `CacheExplainer.explain_performance()`, to indicate whether the marginal distributions (`link_variables == False`) or the whole distribution (`link_variables == True`) is stressed.
* `constraints` parameter to `CacheExplainer.explain_influence()` and `CacheExplainer.explain_performance()`, to specify fixed means for certain features.
* `ethik.query.Query()` class to build queries.

### Changed

### Fixed

* `BaseExplainer.compute_weights()` can now be called with an unamed `pd.Series()` (and so `BaseExplainer.compute_distributions()`).
* Fix numeric noise in `Query.from_taus()`, which now always returns exactly the same targets, independently from whether the feature is alone in the dataset or not.

## 0.0.3 - 15/11/19

### Added

* Added the Pima indians diabetes dataset via the `datasets.load_diabetes` function.
* Added the UCI heart disease dataset via the `datasets.load_heart_disease` function.
* A `ConstantWarning` will now be raised when a feature has only one single value.
* `BaseExplainer.compute_distributions()` and `BaseExplainer.plot_distributions()` to compute and visualize stressed distributions.
* `BaseExplainer.compare_influence()` and `BaseExplainer.compare_performance()` to visualize how the model behaves on average for two individuals. If my friend got the loan and I didn't, this plot tells use which features were responsible of the difference.
* `datasets.load_adult()` to easily load the Adult dataset.

### Changed

* In `ImageClassificationExplainer`, the image plot size will now adapt better to the desired number of rows and columns.
* `Explainer.explain_bias()` renamed into `Explainer.explain_influence()` as "bias" means something specific in statistics.
* Renamed "bias" to "influence" across the library.
* Created a class `BaseExplainer` to do the computations only. `Explainer` was renamed into `CacheExplainer`, inherits from `BaseExplainer` and handles caching and querying (by building a list of taus).
* `value` columns was renamed into `target` in the `.info` to better reflects the paper.

### Fixed

* Fixed an issue where an `IndexError` would be raised if a column name wasn't of type `str`.
* Fixed duplicates when calling `plot_performance()` with multiple labels.
* Fixed convergence issues of the optimization algorithm. Each feature is now standard scaled and we use the `scipy.optimize` module instead of our own procedure.

## 0.0.2 - 02/10/19

### Changed

* Fix bias ranking normalization. Now, we get the minimum and maximum biases
across all features. Before that, a bias curve that spread from 0 to 0.5
and one that spread from 0 to 1 were both normalized to stay between 0 and 1. But
by doing so, we used to loose the information that the second curve was more spread
(and so that the corresponding feature was probably more impactful).
* Loading datasets from the `datasets` submodule should now work correctly.
