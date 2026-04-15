import sys
import secrets
import numpy as np
import pyarrow as pa
from . import _rust

try:
    from sklearn.base import ClassifierMixin as RfClassifierMixin
    from sklearn.base import RegressorMixin as RfRegressorMixin
    from sklearn.base import BaseEstimator as RfBase
    from sklearn.base import MultiOutputMixin as RfMultiOutputMixin
except ImportError:
    class RfBase:
        """Dummy class for sklearn.base.BaseEstimator."""

    class RfClassifierMixin:
        """Dummy class for sklearn.base.ClassifierMixin."""

    class RfRegressorMixin:
        """Dummy class for sklearn.base.ClassifierMixin."""
    
    class RfMultiOutputMixin:
        """Dummy class for sklearn.base.MultiOutputMixin"""

class ResultTable:
    def __init__(self, obj, to_pycapsule):
        if not hasattr(obj, "__arrow_c_stream__"):
            raise AttributeError("Object does not have arrow stream PyCapsule")
        
        self.obj = obj
        self.to_pycapsule = to_pycapsule

    def __arrow_c_stream__(self, requested_schema=None):
        return self.obj.__arrow_c_stream__()

    def get_df(self):
        if not self.to_pycapsule:
            return self._df_to_numpy()
        return self

    def _df_to_numpy(self):
        cols = [self.obj[col].to_numpy(zero_copy_only=False) for col in self.obj.column_names]
        return np.column_stack(cols)


class ImportanceResultTable:
    IMPORTANCE_COL = "importance"
    
    def __init__(self, obj, to_pycapsule):
        if not hasattr(obj, "__arrow_c_stream__"):
            raise AttributeError("Object does not have arrow stream PyCapsule")
        
        self.obj = obj
        self.to_pycapsule = to_pycapsule

    def __arrow_c_stream__(self, requested_schema=None):
        return self.obj.__arrow_c_stream__()

    def get_array(self):
        if not self.to_pycapsule:
            return self._df_to_numpy()
        return self

    def _df_to_numpy(self):
        return self.obj[self.IMPORTANCE_COL].to_numpy(zero_copy_only=False)


class ResultArray:
    def __init__(self, obj, to_pycapsule):
        if not hasattr(obj, "__arrow_c_array__"):
            raise AttributeError("Object does not have arrow array PyCapsule")

        # Pandas does not handle uint64 categoricals, so we need to downcast to uint32.
        # Safe to do as we do not expect millions of classes in classification.
        if pa.types.is_dictionary(obj.type) and pa.types.is_uint64(obj.type.index_type):
            new_indices = obj.indices.cast(pa.uint32())
            obj = pa.DictionaryArray.from_arrays(new_indices, obj.dictionary)
            
        self.obj = obj        
        self.to_pycapsule = to_pycapsule

    def __arrow_c_array__(self, requested_schema=None):
        return self.obj.__arrow_c_array__()

    def get_array(self):
        if not self.to_pycapsule:                
            return self._array_to_numpy()
        return self

    def _array_to_numpy(self):
        return self.obj.to_numpy(zero_copy_only=False)


class RandomForestBase(RfMultiOutputMixin, RfBase):
    def __init__(self, n_estimators, tries=None, save_forest=True, calculate_importance=False, calculate_oob=False, importance_normalised=False, random_state=None, n_jobs=None, to_pycapsule=False):
        self.n_estimators = n_estimators
        self.tries = tries
        self.save_forest = save_forest
        self.calculate_importance = calculate_importance
        self.calculate_oob = calculate_oob
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_normalised = importance_normalised
        self.to_pycapsule = to_pycapsule
        self.forest = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.forest is not None:
            state["forest_bytes"] = self.forest.to_bytes()
            del state["forest"]
        return state
    
    def __setstate__(self, state):
        if "forest_bytes" in state:
            forest = _rust.RandomForest.from_bytes(state["forest_bytes"])
            self.forest = forest
            del state["forest_bytes"]        
        self.__dict__.update(state)
                
    def predict(self, X):
        X = self._validate_x(X)
        preds = self.forest.predict(X, self._get_seed(), self.n_jobs)
        return ResultArray(preds, to_pycapsule=self.to_pycapsule).get_array()

    def oob(self):
        oob = self.forest.oob(self._get_seed())
        return ResultArray(oob, to_pycapsule=self.to_pycapsule).get_array()

    def importance(self):
        importance = self.forest.importance(self.importance_normalised)
        return ImportanceResultTable(importance, to_pycapsule=self.to_pycapsule).get_array()

    def _get_seed(self):
        return self.random_state if self.random_state is not None else secrets.randbits(64)

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


class RandomForestClassifier(RandomForestBase, RfClassifierMixin):
    def fit(self, X, y):
        y = self._validate_y(y)
        X = self._validate_x(X)
        self.forest = _rust.RandomForest(X, y, self.n_estimators, self.tries, self.save_forest, self.calculate_importance, self.calculate_oob, self._get_seed(), True, self.n_jobs)

    def predict_proba(self, X):
        X = self._validate_x(X)
        votes = self.forest.predict_votes(X, self.n_jobs)
        return ResultTable(votes, to_pycapsule=self.to_pycapsule).get_df()

    def oob_votes(self):
        votes = self.forest.oob_votes()
        return ResultTable(votes, to_pycapsule=self.to_pycapsule).get_df()

        
class RandomForestRegressor(RandomForestBase, RfRegressorMixin):
    def fit(self, X, y):
        y = self._validate_y(y)
        X = self._validate_x(X)
        self.forest = _rust.RandomForest(X, y, self.n_estimators, self.tries, self.save_forest, self.calculate_importance, self.calculate_oob, self._get_seed(), False, self.n_jobs)

