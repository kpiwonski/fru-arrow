# Fru-arrow

[R version](https://cran.r-project.org/web/packages/fru/index.html) |
[Pyfru docs](https://kpiwonski.github.io/fru-arrow/)

Fru-arrow is a highly performant implementation of the **Random Forest** model. It uses Arrow PyCapsule underneath,
making integration with any library that supports it - ``polars``, ``pandas``, ``pyarrow`` straightforward.
Moreover, it features permutation importance with a novel, highly optimized algorithm.
It can be used for both **classification** and **regression**, as well as out-of-bag predictions.

Fru is anywhere from 10 to a few thousand times faster than scikit implementation of Random Forest.
The difference grows with the number of threads being used.
The plot below shows this difference dependent on the number of threads.

![Compare to scikit](https://raw.githubusercontent.com/kpiwonski/fru-arrow/refs/heads/main/plt_cmp_scikit.png)
