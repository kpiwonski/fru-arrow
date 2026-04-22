import pickle
import pytest
import numpy as np
import pandas as pd

import pyfru

@pytest.fixture
def table_0_1_3ft():
    # Generate random 0s and 1s
    num_rows = 500
    num_cols = 3
    rng = np.random.default_rng(seed=42)
    data = {f"col{i+1}": rng.integers(0, 2, size=num_rows) for i in range(num_cols)}

    df = pd.DataFrame(data)
    y = pd.Categorical(df.iloc[:,0], categories=[0, 1], ordered=False)
    y = y.rename_categories({0: 'No', 1: 'Yes'})
    y = pd.Series(y)
    return (df, y)


@pytest.fixture
def table_0_1_10ft():
    # Generate random 0s and 1s
    num_rows = 500
    num_cols = 10
    data = {f"col{i+1}": np.random.randint(0, 2, size=num_rows) for i in range(num_cols)}
    df = pd.DataFrame(data)
    y = pd.Categorical(df.iloc[:,0], categories=[0, 1], ordered=False)
    y = y.rename_categories({0: 'No', 1: 'Yes'})
    y = pd.Series(y)
    return (df, y)


@pytest.fixture
def table_uniform_3ft():
    num_rows = 500
    columns = ['A', 'B', 'C']

    # Sample data from a uniform distribution between 0 and 1
    data = np.random.uniform(low=0.0, high=1.0, size=(num_rows, len(columns)))

    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    y = df.iloc[:, 0]
    return (df, y)

def test_rf_cls_0_1_3ft_imp(table_0_1_3ft): 
    X, y = table_0_1_3ft
    rf = pyfru.RandomForestClassifier(100, 1, calculate_importance=True, importance_normalised=False, random_state=1)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01

def test_rf_cls_0_1_3ft_imp_pycapsule(table_0_1_3ft): 
    X, y = table_0_1_3ft
    rf = pyfru.RandomForestClassifier(100, 1, calculate_importance=True, importance_normalised=False, random_state=None, to_pycapsule=True)
    rf.fit(X, y)
    imp = rf.importance()
    imp_df = pd.DataFrame.from_arrow(imp)
    assert list(imp_df.columns) == ["column", "importance"]
    assert list(imp_df["column"]) == ["col1", "col2", "col3"]
    assert imp_df["importance"][0] > 0.1
    assert imp_df["importance"][1] < 0.01
    assert imp_df["importance"][2] < 0.01



def test_rf_cls_0_1_3ft_tries_none_imp(table_0_1_10ft): 
    X, y = table_0_1_10ft
    rf = pyfru.RandomForestClassifier(100, None, calculate_importance=True, random_state=1)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01
    
    
def test_rf_cls_0_1_3ft_oob(table_0_1_3ft): 
    X, y = table_0_1_3ft
    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=1)
    rf.fit(X, y)
    assert sum(y.to_numpy() == rf.oob())/len(y) > 0.95

    votes = rf.oob_votes()
    assert sum((votes[:,0] < votes[:,1]) == y.cat.codes.to_numpy()) > 0.95
    
def test_rf_cls_0_1_3ft_predict(table_0_1_3ft): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    assert sum(y_test.to_numpy() == y_pred)/len(y_test) > 0.95
  
    votes = rf.predict_proba(X_test)
    assert sum((votes[:,0] < votes[:,1]) == y_test.cat.codes.to_numpy()) > 0.95


def test_rf_cls_0_1_3ft_predict_validates_column_names(table_0_1_3ft, tmp_path): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    X_test.columns = ["col_1", "col_2", "XXX"]
    y_train = y[:400]
    
    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None)
    rf.fit(X_train, y_train)
    with pytest.raises(BaseException, match="Column names do not match"):
        rf.predict(X_test)
  
    with pytest.raises(BaseException, match="Column names do not match"):
        rf.predict_proba(X_test)

        
    model_path = tmp_path / "model_validate_columns.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    with open(model_path, "rb") as f:
        loaded_rf = pickle.load(f)

    
    with pytest.raises(BaseException, match="Column names do not match"):
        loaded_rf.predict(X_test)
  
    with pytest.raises(BaseException, match="Column names do not match"):
        loaded_rf.predict_proba(X_test)


def test_rf_cls_0_1_3ft_predict_validates_dtypes(table_0_1_3ft): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    print(X_test.dtypes)
    X_test["col1"] = X_test["col1"].astype(float)
    y_train = y[:400]
    
    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None)
    rf.fit(X_train, y_train)
    with pytest.raises(BaseException, match="Column types do not match"):
        rf.predict(X_test)
  
    with pytest.raises(BaseException, match="Column types do not match"):
        rf.predict_proba(X_test)


def test_rf_cls_predict_validates_categorical_unique_values(): 
    x1 = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    x2 = ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X']
    df = pd.DataFrame({'col1': x1, 'col2': x2}, dtype="category")

    y = pd.Series(
        ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        dtype='category'
    )

    x1 = df["col1"]
    x2 = df["col2"]
    df_predict = pd.DataFrame({"col1": x2, "col2": x1}, dtype="category")
    
    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None)
    rf.fit(df, y)
    with pytest.raises(BaseException, match="Categorical features unique values do not match"):
        rf.predict(df_predict)
  
    with pytest.raises(BaseException, match="Categorical features unique values do not match"):
        rf.predict_proba(df_predict)


def test_rf_cls_0_1_3ft_predict_pickle_roundtrip(table_0_1_3ft, tmp_path): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, calculate_importance=True, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    np.testing.assert_array_equal(y_pred, y_test)
    votes = rf.predict_proba(X_test)
    oob_votes = rf.oob_votes()
    importance = rf.importance()

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    with open(model_path, "rb") as f:
        loaded_rf = pickle.load(f)

    y_loaded_pred = loaded_rf.predict(X_test)
    np.testing.assert_array_equal(y_pred, y_loaded_pred)
    loaded_votes = loaded_rf.predict_proba(X_test)
    loaded_oob_votes = loaded_rf.oob_votes()
    np.testing.assert_array_equal(oob_votes, loaded_oob_votes)
    loaded_importance = loaded_rf.importance()
    np.testing.assert_array_equal(importance, loaded_importance)
    
    np.testing.assert_array_equal(votes, loaded_votes)


def test_rf_cls_0_1_3ft_predict_pickle_roundtrip_without_importance_and_oob(table_0_1_3ft, tmp_path): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=False, calculate_importance=False, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    np.testing.assert_array_equal(y_pred, y_test)
    votes = rf.predict_proba(X_test)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    with open(model_path, "rb") as f:
        loaded_rf = pickle.load(f)

    y_loaded_pred = loaded_rf.predict(X_test)
    np.testing.assert_array_equal(y_pred, y_loaded_pred)
    loaded_votes = loaded_rf.predict_proba(X_test)
    np.testing.assert_array_equal(votes, loaded_votes)

def test_rf_cls_0_1_3ft_pickle_without_save_forest(table_0_1_3ft, tmp_path): 
    X, y = table_0_1_3ft

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, calculate_importance=True, save_forest=False, random_state=1)
    rf.fit(X, y)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        with pytest.raises(ValueError, match="Cannot serialize model, when forest is not saved"):
            pickle.dump(rf, f)


def test_rf_reg_importance(table_uniform_3ft):
    X, y = table_uniform_3ft
    rf = pyfru.RandomForestRegressor(100, 1, calculate_importance=True, importance_normalised=False, random_state=1)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01


def test_rf_reg_oob(table_uniform_3ft):
    X, y = table_uniform_3ft
   
    rf = pyfru.RandomForestRegressor(100, 1, calculate_oob=True, random_state=1)
    rf.fit(X, y)
    y_oob = rf.oob()
    assert sum((y_oob-y).abs())/len(y) < 0.05
    

def test_rf_reg_predict(table_uniform_3ft):
    X, y = table_uniform_3ft
    
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]
    
    rf = pyfru.RandomForestRegressor(100, 1, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    assert sum((y_pred-y_test).abs())/len(y_pred) < 0.05


def test_rf_res_to_pandas(table_0_1_3ft):
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, calculate_importance=True, random_state=None, to_pycapsule=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    assert len(pd.Series.from_arrow(y_pred)) == 100
    assert len(pd.Series.from_arrow(rf.oob())) == 400
    assert pd.DataFrame.from_arrow(rf.importance()).shape[0] == 3
    assert pd.DataFrame.from_arrow(rf.predict_proba(X_test)).shape == (100, 2)
    assert pd.DataFrame.from_arrow(rf.oob_votes()).shape == (400, 2)
    
        
