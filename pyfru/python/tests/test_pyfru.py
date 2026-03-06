import pytest
import numpy as np
import pandas as pd

import pyfru

@pytest.fixture
def table_0_1_3ft():
    # Generate random 0s and 1s
    num_rows = 500
    num_cols = 3
    data = {f"col{i+1}": np.random.randint(0, 2, size=num_rows) for i in range(num_cols)}

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
    rf = pyfru.RandomForestClassifier(100, 1, calculate_importance=True, importance_normalised=False, random_state=None)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01


def test_rf_cls_0_1_3ft_tries_none_imp(table_0_1_10ft): 
    X, y = table_0_1_10ft
    rf = pyfru.RandomForestClassifier(100, None, calculate_importance=True, random_state=None)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01
    
    
def test_rf_cls_0_1_3ft_oob(table_0_1_3ft): 
    X, y = table_0_1_3ft
    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None)
    rf.fit(X, y)
    assert sum(y.cat.codes.to_numpy() == rf.oob())/len(y) > 0.95

    votes = rf.oob_votes
    assert sum((votes[:,0] < votes[:,1]) == y.cat.codes.to_numpy()) > 0.95
    
def test_rf_cls_0_1_3ft_predict(table_0_1_3ft): 
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    assert sum(y_test.cat.codes.to_numpy() == y_pred)/len(y_test) > 0.95
  
    votes = rf.predict_proba(X_test)
    assert sum((votes[:,0] < votes[:,1]) == y_test.cat.codes.to_numpy()) > 0.95


def test_rf_reg_importance(table_uniform_3ft):
    X, y = table_uniform_3ft
    rf = pyfru.RandomForestRegressor(100, 1, calculate_importance=True, importance_normalised=False, random_state=None)
    rf.fit(X, y)
    imp = rf.importance()
    assert imp[0] > 0.1
    assert imp[1] < 0.01
    assert imp[2] < 0.01


def test_rf_reg_oob(table_uniform_3ft):
    X, y = table_uniform_3ft
   
    rf = pyfru.RandomForestRegressor(100, 1, calculate_oob=True, random_state=None)
    rf.fit(X, y)
    y_oob = rf.oob()
    assert sum((y_oob-y).abs())/len(y) < 0.05
    

def test_rf_reg_predict(table_uniform_3ft):
    X, y = table_uniform_3ft
    
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]
    y_test = y[400:]
    
    rf = pyfru.RandomForestRegressor(100, 1, random_state=None)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    assert sum((y_pred-y_test).abs())/len(y_pred) < 0.05


def test_rf_res_to_pandas(table_0_1_3ft):
    X, y = table_0_1_3ft
    X_train = X.iloc[:400,:]
    X_test = X.iloc[400:,:]
    y_train = y[:400]

    rf = pyfru.RandomForestClassifier(100, 1, calculate_oob=True, random_state=None, to_pycapsule=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    assert len(pd.Series.from_arrow(y_pred)) == 100
    assert len(pd.Series.from_arrow(rf.oob())) == 400
    assert pd.DataFrame.from_arrow(rf.importance()).shape[0] == 3
    assert pd.DataFrame.from_arrow(rf.predict_proba(X_test)).shape == (100, 2)
    assert pd.DataFrame.from_arrow(rf.oob_votes).shape == (400, 2)
    
        
