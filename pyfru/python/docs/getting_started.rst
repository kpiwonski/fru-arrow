Getting started
===============


Installation
------------
The process is similar to installing other Python packages. Use your preferred tool — ``pip``, ``poetry``, or ``uv``.

.. code-block:: bash

    mkdir test_pyfru && cd test_pyfru
    python -m venv .venv
    source .venv/bin/activate
    pip install scikit-learn pyfru

    # Optional: install additional libraries for data handling
    pip install pandas


Basic usage example
-------------------
The package provides a scikit-learn-like API. You can initialize a model and use ``fit`` and ``predict`` in a similar way.
For most users, the only parameter that needs adjustment is ``trees``, which specifies the number of trees.

.. warning::
    Pandas fully supports the PyCapsule interface starting from version 3.
    We support pandas version 3 and above.

.. warning::
    The target must be a categorical series with string categories. Passing NumPy arrays is not supported.
    If you are using NumPy arrays, convert them to ``pandas.DataFrame`` and ``pandas.Series``.
	
.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    from pyfru import RandomForestClassifier	

    # load data
    data = load_iris(as_frame=True)
	
    # convert target to a categorical series with string categories
    y = data["target"].astype(str).astype("category")
	
    X_train, X_test, y_train, y_test = train_test_split(data["data"], y, test_size=0.2)
	
    # create model instance
    rf = RandomForestClassifier(trees=100)

    # fit model
    rf.fit(X_train, y_train)

    # make predictions
    rf.predict(X_test)

Permutation importance and out-of-bag
-------------------------------------
One of the key features of this model is support for permutation importance and out-of-bag (OOB) estimation.
These features must be enabled when initializing the model by setting ``calculate_importance=True`` and ``oob=True``.

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    from pyfru import RandomForestClassifier

    # load data
    data = load_iris(as_frame=True)
    X = data["data"]
    y = data["target"].astype(str).astype("category")

    # create model instance with additional features enabled
    rf = RandomForestClassifier(trees=100, calculate_importance=True, oob=True)
    # fit model
    rf.fit(X, y)

    # get permutation importance	
    rf.importance()

    # make predictions
    rf.predict()

    # get predictions with voting details	
    rf.predict(votes=True)


Regressor model
---------------
The package also supports regression. The target value can be any numeric type.
Internally, it is converted to ``float64``, and predictions are returned as ``float64`` as well.

.. code-block:: python
 
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes

    from pyfru import RandomForestRegressor

    # load data
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    # create model instancec
    rf = RandomForestRegressor(trees=100, calculate_importance=True, oob=True)

    # fit model	
    rf.fit(X, y)

    # get permutation importance
    rf.importance()

    # get out-of-bag predictions	
    rf.predict()


Polars example
--------------

Another feature of ``Pyfru`` is that it works with libraries supporting the Arrow PyCapsule interface,
such as ``pandas``, ``polars``, ``pyarrow``, ``duckdb``, and others.

.. code-block:: python

    import polars as pl

    from sklearn.datasets import load_iris

    from pyfru import RandomForestClassifier	

    # load data	
    data = load_iris(as_frame=True)

    # create model instance
    x = pl.from_pandas(data["data"])
    y = pl.from_pandas(data["target"].astype(str).astype("category"))

    # create model instance
    rf = RandomForestClassifier(trees=100, calculate_importance=True, oob=True)

    # fit model
    rf.fit(X, y)

    # get permutation importance	
    rf.importance()

    # get out-of-bag predictions
    rf.predict()


Result as PyCapsule (optional)
------------------------------
Results can optionally be returned as an Arrow PyCapsule. This allows them to be
loaded into any data frame library supporting the Arrow PyCapsule interface.

.. code-block:: python

    import pandas as pd

    from sklearn.datasets import load_iris

    from pyfru import RandomForestClassifier

    # load data
    data = load_iris(as_frame=True)
    X = data["data"]
    y = data["target"].astype(str).astype("category")
	
    # create model instance	
	rf = RandomForestClassifier(trees=100, calculate_importance=True, oob=True)

    # fit model	
    rf.fit(X, y)

    # convert results from PyCapsule	
    pd.DataFrame.from_arrow(rf.importance(to_pycapsule=True))
    pd.Series.from_arrow(rf.predict(to_pycapsule=True))
    pd.DataFrame.from_arrow(rf.predict(votes=True, to_pycapsule=True))


Serialization
---------------

The model can be serialized using ``pickle``.

.. code-block:: python

    import pickle

    from sklearn.datasets import load_iris

    from pyfru import RandomForestClassifier


    # load data
    data = load_iris(as_frame=True)
    X = data["data"]
    y = data["target"].astype(str).astype("category")
	
    # create model instance
    rf = RandomForestClassifier(trees=100, calculate_importance=True, oob=True)

    # fit model	
    rf.fit(X, y)

    # save model
    with open("rf.pkl", "wb") as f:
        pickle.dump(rf, f)

    # load model
    with open("rf.pkl", "rb") as f:
        loaded_rf = pickle.load(f)

    # use loaded model
    loaded_rf.predict()
