# Changelog

## Unreleased

## [0.0.2] - 2019-10-02

### Fixed

* Fix bias ranking normalization. Now, we get the minimum and maximum biases
across all features. Before that, a bias curve that spread from 0 to 0.5
and one that spread from 0 to 1 were both normalized to stay between 0 and 1. But
by doing so, we used to loose the information that the second curve was more spread
(and so that the corresponding feature was probably more impactful).
* Loading datasets from the `datasets` submodule should now work correctly.
