# Fru-arrow

[R version](https://cran.r-project.org/web/packages/fru/index.html) |
[Pyfru docs](https://kpiwonski.github.io/fru-arrow/)

Fru-arrow is a highly performant implementation of the **Random Forest** model. It uses Arrow PyCapsule underneath,
making integration with any library that supports it - ``polars``, ``pandas``, ``pyarrow`` straightforward.
Moreover, it features permutation importance with a novel, highly optimized algorithm.
It can be used for both **classification** and **regression**, as well as out-of-bag predictions.

Fru is typically 10 to several thousand times faster than the scikit-learn implementation of Random Forest.
The gap increases as you use more threads.

The plot below shows how the difference scales with the number of threads on the Iris dataset.

![Compare to scikit](https://raw.githubusercontent.com/kpiwonski/fru-arrow/refs/heads/main/plt_cmp_scikit.png)
