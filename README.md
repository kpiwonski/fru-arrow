# Fru-arrow

[![PyPI Version](https://img.shields.io/pypi/v/pyfru)](https://pypi.org/project/pyfru/)
[![Crates.io Version](https://img.shields.io/crates/v/fru-arrow)](https://crates.io/crates/fru-arrow)

[Link to preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6864991) |
[Pyfru docs](https://kpiwonski.github.io/fru-arrow/) |
[R version](https://cran.r-project.org/web/packages/fru/index.html)

Fru-arrow is a highly performant implementation of the **Random Forest** model. It uses Arrow PyCapsule underneath,
making integration with any library that supports it - ``polars``, ``pandas``, ``pyarrow`` straightforward.
Moreover, it features permutation importance with a novel, highly optimized algorithm.
It can be used for both **classification** and **regression**, as well as out-of-bag predictions.

Fru is typically anywhere from a few times to several thousand times faster than scikit-learn's Random Forest implementation.
The performance gap widens as the number of threads increases.

The plot below illustrates this difference.

![Compare to scikit](https://raw.githubusercontent.com/kpiwonski/fru-arrow/refs/heads/main/plt_cmp_scikit.png)
