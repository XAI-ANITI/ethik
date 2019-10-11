# Changelog

## Unreleased

### Added

* Added the Pima indians diabetes dataset via the `datasets.load_diabetes` function.
* Added the UCI heart disease dataset via the `datasets.load_heart_disease` function.

### Changed

* In `ImageClassificationExplainer`, the image plot size will now adapt better to the desired number of rows and columns.

## 0.0.2 - 02/10/19

### Changed

* Fix bias ranking normalization. Now, we get the minimum and maximum biases
across all features. Before that, a bias curve that spread from 0 to 0.5
and one that spread from 0 to 1 were both normalized to stay between 0 and 1. But
by doing so, we used to loose the information that the second curve was more spread
(and so that the corresponding feature was probably more impactful).
* Loading datasets from the `datasets` submodule should now work correctly.
