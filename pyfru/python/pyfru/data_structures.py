import numpy as np
import pyarrow as pa

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

