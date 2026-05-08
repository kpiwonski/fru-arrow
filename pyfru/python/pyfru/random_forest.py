from abc import ABC
from pyfru.data_structures import ImportanceResultTable, ResultArray, ResultTable
import sys
import secrets

from . import _rust


class _RandomForestBase(ABC):
    def __init__(self, trees=500, tries=None, save_forest=True, calculate_importance=False, oob=False, seed=None, threads=None):
        self.trees = trees
        self.tries = tries
        self.save_forest = save_forest
        self.calculate_importance = calculate_importance
        self.oob = oob
        self.seed = seed
        self.threads = threads
        self._forest = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._forest is not None:
            state["forest_bytes"] = self._forest.to_bytes()
            del state["_forest"]
        return state
    
    def __setstate__(self, state):
        if "forest_bytes" in state:
            forest = _rust.RandomForest.from_bytes(state["forest_bytes"])
            self._forest = forest
            del state["forest_bytes"]        
        self.__dict__.update(state)

    def importance(self, scale=False, to_pycapsule=False):
        """
        Extract permutation (mean decrease in accuracy) importance from the model.

        Parameters
        ----------
        scale : bool
            If ``True``, values are scaled by their standard deviation over the ensemble.
            Defaults to ``False``.
        to_pycapsule : bool
            If ``True``, results are returned as an Arrow PyCapsule. If ``False``, results
            are returned as NumPy arrays, similar to scikit-learn. Defaults to ``False``.

        Returns
        -------
        numpy.ndarray or Arrow PyCapsule
            By default, a 1D NumPy array is returned. It contains the importance
            value for each column. If ``to_pycapsule=True``, an object implementing
            the Arrow PyCapsule interface is returned. This object can be loaded as
            a DataFrame in libraries supporting the Arrow PyCapsule interface. It
            has two columns: ``column`` and ``importance``.

        Notes
        -----
        Other packages often scale importance by its standard error estimate,
        producing importance values larger by a factor of the square root of
        the number of trees compared to fru.
        """
        importance = self._forest.importance(scale)
        return ImportanceResultTable(importance, to_pycapsule=to_pycapsule).get_array()

    def _get_seed(self):
        return self.seed if self.seed is not None else secrets.randbits(64)

    @staticmethod
    def _remove_pandas_rownames(obj):
        pd = sys.modules.get("pandas")
    
        # Remove pandas rownames, otherwise would be passed as __index_level_0__ to arrow
        if pd and isinstance(obj, pd.DataFrame):
            return obj.reset_index(drop=True)
        return obj
   
    @classmethod
    def _validate_x(cls, X):
        if not hasattr(X, "__arrow_c_stream__"):
            raise AttributeError("X must implement PyCapsule")
        X = cls._remove_pandas_rownames(X)
        return X
    
    @staticmethod
    def _validate_y(y):
        if not hasattr(y, "__arrow_c_stream__"):
            raise AttributeError("y must implement PyCapsule")
        return y


class RandomForestClassifier(_RandomForestBase):
    """
        Random Forest Classifier implementing a fast multi-thread implementation of
        the original Random Forest model [1]_. Thanks to the Arrow PyCapsule interface,
        Pyfru supports a wide range of DataFrame libraries. In addition to prediction,
        it supports out-of-bag (OOB) predictions and feature importance. The importance
        measure is computed in a novel, efficient way.

        Parameters
        ----------
        trees : int
            Number of trees to grow in the forest (often called ``ntree`` in other
            software). Must be greater than zero. The value should be large enough
            to provide stable results (prediction accuracy or importance). Larger
            datasets typically require more trees. Computation time grows linearly
            with the number of trees. Defaults to 500.
        tries : int | None
            Number of features to try at each split (often called ``mtry``).
            Must be greater than zero and less than or equal to the number of features.
            By default, it is set to the rounded square root of the number of features.
            Higher values increase correlation between trees. In most cases, the default
            setting is recommended.
        save_forest : bool
            If ``True``, the fitted forest is stored and can be used for prediction
            or serialization. Set to ``False`` when only importance or OOB results
            are needed. Defaults to ``True``.
        calculate_importance : bool
            If ``True``, feature importance is calculated. Defaults to ``False``.
        oob : bool
            If ``True``, out-of-bag predictions are calculated. Defaults to ``True``.
        seed : int | None
            Seed used by the algorithm. Set to ``None`` to use a random seed.        
        threads : int | None
            Number of threads to use. Must be greater than zero. If ``None``, all
            available CPU cores are used. Defaults to ``None``.
        Notes
        -----
        The Random Forest Classifier selects the best feature at each split using
        Gini impurity. For numerical features, the threshold is optimized by an
        exhaustive scan. The threshold is the midpoint between
        adjacent values. In case of ties, the smaller threshold is chosen.
        
        Ordered categorical variables are currently treated as categorical. Variables
        with six or more levels are treated as integers. Variables with five or fewer
        levels are split by exhaustively evaluating all possible partitions into two
        subsets using the Gini criterion.       
        
        The maximum tree depth is fixed at 512. A leaf is created when the sample size
        reaches one. Leaves may also be formed earlier if no valid split is found; in
        such cases, ties are broken randomly.
        
        Pyfru uses its own PRNG, the pcg32 method by Melissa E. O'Neill [2]_, to provide
        reproducible results in parallel settings. For a given input and seed, the same
        trees are built regardless of the number of threads, although their order may
        differ. OOB predictions and importance scores are therefore consistent up to
        numerical precision.

        References
        ----------
        .. [1] `L. Breiman, Random Forests, Machine Learning, 45(1), 5-32, 2001.
           <https://doi.org/10.1023/A:1010933404324>`_
        .. [2] `O'Neil Melissa E. (2014). PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation.
           <https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf>`_
    """

    def fit(self, X, y):
        """
        Build a forest of trees from the training data.
    
        Parameters
        ----------
        X : Arrow PyCapsule
            A DataFrame-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used (e.g., pandas,
            polars). Columns can be boolean, numerical, or categorical. Mixed
            column types are allowed. Other data types are not supported and
            will raise an exception. ``NaN`` values are not allowed.
        y : Arrow PyCapsule
            A Series-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used. For
            classification, ``y`` must be categorical, otherwise an exception
            is raised. ``NaN`` values are not allowed. The length of ``y`` must
            match the number of rows in ``X``.            
        """
        y = self._validate_y(y)
        X = self._validate_x(X)
        self._forest = _rust.RandomForest(X, y, self.trees, self.tries, self.save_forest, self.calculate_importance, self.oob, self._get_seed(), True, self.threads)

    def predict(self, X=None, votes=False, validate_input=True, to_pycapsule=False):
        """
        Predict for X. The predicted class is the one with the highest number
        of votes across the trees. If no X is given, out-of-bag predictions
        are returned. If ``votes=True``, vote counts are returned instead of
        predictions.

        Parameters
        ----------
        X : Arrow PyCapsule | None
            A DataFrame-like object supporting the Arrow PyCapsule interface.
            Cannot contain ``NaN`` values. If ``None`` (default), out-of-bag
            predictions are returned.
        votes: bool
            If ``True``, the number of votes from each tree is returned instead
            of predictions. Defaults to ``False``.
        validate_input : bool
            If ``True``, column data types and names in ``X`` are checked against
            the data provided to ``fit``. For categorical columns, category
            names and their order must match.
        to_pycapsule : bool, optional
            If ``True``, results are returned as an Arrow PyCapsule. Otherwise,
            NumPy arrays are returned. Defaults to ``False``.           
            
        Returns
        -------
        numpy.ndarray or Arrow PyCapsule
            For predictions, a 1D NumPy array is returned with one prediction
            per observation. Categorical predictions are represented as strings.
            For votes, a 2D NumPy array is returned. If ``to_pycapsule=True``,
            results are returned as an Arrow PyCapsule that can be loaded as a
            Series in libraries supporting this interface.
            
        Notes
        -----
        Voting in classification may result in ties. In such cases, a PRNG is
        used to break ties. If ``seed=None``, different seeds are used on each
        execution, so ties may be resolved differently. For deterministic
        behavior, set ``seed`` or use ``votes=True`` to inspect vote counts.

        This method checks that the input structure matches the training data
        structure stored in the object. This may take time for large data or
        low-latency use cases. Set ``validate_input=False`` to skip validation
        and assume the input is correct.
        """
        if votes == False:
            if X is None:
                preds = self._forest.oob(self._get_seed())
            else:
                X = self._validate_x(X)
                preds = self._forest.predict(X, self._get_seed(), validate_input, self.threads)
            return ResultArray(preds, to_pycapsule=to_pycapsule).get_array()

        if X is None:
            votes = self._forest.oob_votes()
        else:
            X = self._validate_x(X)
            votes = self._forest.predict_votes(X, validate_input, self.threads)
        return ResultTable(votes, to_pycapsule=to_pycapsule).get_df()
        

class RandomForestRegressor(_RandomForestBase):
    """
        Random Forest Regressor implementing a fast multi-threaded version of the
        original Random Forest model [3]_. Thanks to the Arrow PyCapsule interface,
        Pyfru supports a wide range of DataFrame libraries. In addition to prediction,
        it supports out-of-bag (OOB) predictions and feature importance. The importance
        measure is computed in a novel, efficient way.

        Parameters
        ----------
        trees : int
            Number of trees to grow in the forest (often called ``ntree`` in other
            software). Must be greater than zero. The value should be large enough
            to provide stable results (prediction accuracy or importance). Larger
            datasets typically require more trees. Computation time grows linearly
            with the number of trees. Defaults to 500.
        tries : int | None
            Number of features to try at each split (often called ``mtry``).
            Must be greater than zero and less than or equal to the number of features.
            By default, it is set to the rounded square root of the number of features.
            Higher values increase correlation between trees. In most cases, the default
            setting is recommended.
        save_forest : bool
            If ``True``, the fitted forest is stored and can be used for prediction
            or serialization. Set to ``False`` when only importance or OOB results
            are needed. Defaults to ``True``.
        calculate_importance : bool
            If ``True``, feature importance is calculated. Defaults to ``False``.
        oob : bool
            If ``True``, out-of-bag predictions are calculated. Defaults to ``True``.
        seed : int | None
            Seed used by the algorithm. Set to ``None`` to use a random seed.        
        threads : int | None
            Number of threads to use. Must be greater than zero. If ``None``, all
            available CPU cores are used. Defaults to ``None``.

        Notes
        -----
        The Random Forest Regressor selects the best feature at each split using
        variance reduction. For numerical features, the threshold is optimized by
        an exhaustive scan. For float features, the threshold is the midpoint between
        adjacent values; for integer features, it is the smaller of the two values.
        In case of ties, the smaller threshold is chosen.

        Ordered categorical variables are currently treated as categorical. Variables
        with six or more levels are treated as integers. Variables with five or fewer
        levels are split by exhaustively evaluating all possible partitions into two
        subsets using the variance reduction criterion.       

        The maximum tree depth is fixed at 512. A leaf is created when the sample size
        reaches four. This means regression typically requires at least ten samples
        to be practical.

        Pyfru uses its own PRNG, the pcg32 method by Melissa E. O'Neill [4]_, to provide
        reproducible results in parallel settings. For a given input and seed, the same
        trees are built regardless of the number of threads, although their order may
        differ. OOB predictions and importance scores are therefore consistent up to
        numerical precision.

        References
        ----------
        .. [3] `L. Breiman, Random Forests, Machine Learning, 45(1), 5-32, 2001.
           <https://doi.org/10.1023/A:1010933404324>`_
        .. [4] `O'Neil Melissa E. (2014). PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation.
           <https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf>`_
    """
    
    def fit(self, X, y):
        """
        Build a forest of trees from the training set.

        Parameters
        ----------
        X : Arrow PyCapsule
            A DataFrame-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used (e.g., pandas,
            polars). Columns can be boolean, numerical, or categorical. Mixed
            column types are allowed. Other data types are not supported and
            will raise an exception. ``NaN`` values are not allowed.
        y : Arrow PyCapsule
            A Series-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used. For regression,
            ``y`` must be of type float or int; otherwise an exception is raised.
            ``NaN`` values are not allowed. The length of ``y`` must match the
            number of rows in ``X``.
        """
        y = self._validate_y(y)
        X = self._validate_x(X)
        self._forest = _rust.RandomForest(X, y, self.trees, self.tries, self.save_forest, self.calculate_importance, self.oob, self._get_seed(), False, self.threads)

    def predict(self, X=None, validate_input=True, to_pycapsule=False):
        """
        Predict for X using the mean over the ensemble.
        If no X is given, out-of-bag predictions are returned.

        Parameters
        ----------
        X : Arrow PyCapsule | None
            A DataFrame-like object supporting the Arrow PyCapsule interface.
            Cannot contain ``NaN`` values. If ``None`` (default), out-of-bag
            predictions are returned.
        validate_input : bool
            If ``True``, column data types and names in ``X`` are checked against
            the data provided to ``fit``. For categorical columns, category
            names and their order must match.
        to_pycapsule : bool, optional
            If ``True``, results are returned as an Arrow PyCapsule. Otherwise,
            NumPy arrays are returned. Defaults to ``False``.           

        Returns
        -------
        numpy.ndarray or Arrow PyCapsule
            For predictions, a 1D NumPy array is returned containing one prediction
            per observation. If ``to_pycapsule=True``, results are returned as an
            Arrow PyCapsule, which can be loaded as a Series in libraries supporting
            this interface.

        Notes
        -----
        Regression is performed using leaf averages, which is deterministic, aside
        from potential small numerical differences due to multithreading and tree
        construction order.

        This method checks that the input structure matches the training data
        structure stored in the object. This may take time for large data or
        low-latency use cases. Set ``validate_input=False`` to skip validation
        and assume the input is correct.
        """
        if X is None:
            preds = self._forest.oob(self._get_seed())
        else:
            X = self._validate_x(X)
            preds = self._forest.predict(X, self._get_seed(), validate_input, self.threads)
        return ResultArray(preds, to_pycapsule=to_pycapsule).get_array()

