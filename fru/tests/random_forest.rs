use fru::RandomForest;
use minarrow::{
    Array, BooleanArray, FieldArray, FloatArray, IntegerArray, RowSelection, Table, TextArray,
};
use minarrow::{CategoricalArray, NumericArray};
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
const NROW: usize = 100;

fn sample_0_1(rng: &mut impl Rng, k: usize) -> Vec<i64> {
    (0..k)
        .map(|_| rng.random_range(0..=1))
        .collect::<Vec<i64>>()
}

fn new_arr_i64(name: &str, x: Vec<i64>) -> FieldArray {
    FieldArray::from_arr(name, Array::from_int64(IntegerArray::<i64>::from_slice(&x)))
}

fn new_arr_f64(name: &str, x: Vec<f64>) -> FieldArray {
    FieldArray::from_arr(name, Array::from_float64(FloatArray::<f64>::from_slice(&x)))
}

fn new_arr_bool(name: &str, x: Vec<bool>) -> FieldArray {
    FieldArray::from_arr(name, Array::from_bool(BooleanArray::<()>::from_slice(&x)))
}

fn new_arr_categorical64(name: &str, x: Vec<u64>, dict: &[String]) -> FieldArray {
    FieldArray::from_arr(
        name,
        Array::from_categorical64(CategoricalArray::<u64>::from_slices(&x, dict)),
    )
}

#[test]
fn rf_cls_check_0_1_3ft() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, NROW);
    let y: Vec<u64> = x1.iter().map(|&x| x as u64).collect();

    let a1 = new_arr_i64("x1", x1);
    let a2 = new_arr_i64("x2", sample_0_1(&mut rng, NROW));
    let a3 = new_arr_i64("x3", sample_0_1(&mut rng, NROW));

    let unique_values = vec![String::from("false"), String::from("true")];

    let df_vec = vec![a1, a2, a3];
    let rf = RandomForest::fit(
        Table::new("table".into(), df_vec.into()),
        Array::from_categorical64(CategoricalArray::from_slices(&y, &unique_values)),
        100,
        1,
        false,
        true,
        false,
        1,
        None,
    );
    let imp_table = rf.importance(false);

    let imp = imp_table.cols[imp_table.col_name_index("importance").unwrap()]
        .array
        .inner::<FloatArray<f64>>();

    assert!(imp[0] > 0.2);
    assert!(imp[1] < 0.01);
    assert!(imp[2] < 0.01);
}

#[test]
fn rf_cls_importance_0_1_interactions() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, 100);
    let x2 = sample_0_1(&mut rng, 100);

    let y_ins: Vec<_> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u64)
        .collect();
    let unique_values = vec![String::from("false"), String::from("true")];

    let mut df_vec = vec![new_arr_i64("x1", x1), new_arr_i64("x2", x2)];
    for i in 1..100 {
        df_vec.push(new_arr_i64(
            &format!("rand{}", i),
            sample_0_1(&mut rng, NROW),
        ));
    }

    let rf = RandomForest::fit(
        Table::new("table".into(), df_vec.into()),
        Array::from_categorical64(CategoricalArray::from_slices(&y_ins, &unique_values)),
        1000,
        10,
        false,
        true,
        false,
        1,
        None,
    );
    let imp_table = rf.importance(true);

    let imp = imp_table.cols[imp_table.col_name_index("importance").unwrap()]
        .array
        .inner::<FloatArray<f64>>();

    assert!(imp[0] > 0.5);
    assert!(imp[1] > 0.5);
    for i in 2..imp.len() {
        assert!(imp[i] < 0.2);
    }
}

#[test]
fn rf_cls_oob_0_1_interactions() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, NROW);
    let x2 = sample_0_1(&mut rng, NROW);

    let y_ins: Vec<_> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u64)
        .collect();

    let mut df_vec = vec![new_arr_i64("x1", x1), new_arr_i64("x2", x2)];
    for i in 1..10 {
        df_vec.push(new_arr_i64(
            &format!("rand{}", i),
            sample_0_1(&mut rng, NROW),
        ));
    }
    let unique_values = vec![String::from("false"), String::from("true")];
    let y = CategoricalArray::from_slices(&y_ins, &unique_values);
    let yy = y.clone();

    let rf = RandomForest::fit(
        Table::new("table".into(), df_vec.into()),
        Array::from_categorical64(y),
        1000,
        3,
        false,
        false,
        true,
        1,
        None,
    );

    let oob_pred = rf.oob(1);
    assert!(matches!(
        oob_pred,
        Array::TextArray(TextArray::Categorical64(_))
    ));

    let Array::TextArray(TextArray::Categorical64(arr)) = oob_pred else {
        unreachable!()
    };

    let score = arr
        .iter()
        .zip(yy.iter())
        .map(|(&x, &y)| (x == y) as u64)
        .sum::<u64>();
    assert!(score == 100);

    let pred = rf.oob_votes();

    let score = pred.cols[0]
        .array
        .inner::<IntegerArray<u64>>()
        .iter()
        .zip(pred.cols[1].array.inner::<IntegerArray<u64>>().iter())
        .zip(yy.iter())
        .map(|((&x1, &x2), &y)| ((x2 as f64 / (x1 + x2) as f64 > 0.5) as u64 == y) as u64)
        .sum::<u64>();

    assert!(score == 100);
}

#[test]
fn rf_cls_predict_0_1_interactions() {
    let nrow = 200;
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, nrow);
    let x2 = sample_0_1(&mut rng, nrow);

    let y_ins: Vec<_> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u64)
        .collect();

    let mut df_vec = vec![new_arr_i64("x1", x1), new_arr_i64("x2", x2)];
    for i in 1..10 {
        df_vec.push(new_arr_i64(
            &format!("rand{}", i),
            sample_0_1(&mut rng, nrow),
        ));
    }
    let unique_values = vec![String::from("false"), String::from("true")];
    let y = CategoricalArray::from_slices(&y_ins[0..100], &unique_values);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df.r(0..100).to_table(),
        Array::from_categorical64(y),
        1000,
        3,
        true,
        false,
        false,
        1,
        None,
    );

    let oob_pred = rf.predict(df.r(100..200).to_table(), 1, None);
    assert!(matches!(
        oob_pred,
        Array::TextArray(TextArray::Categorical64(_))
    ));

    let Array::TextArray(TextArray::Categorical64(arr)) = oob_pred else {
        unreachable!()
    };
    let score = arr
        .iter()
        .zip(y_ins[100..200].iter())
        .map(|(&x, &y)| (x == y) as u64)
        .sum::<u64>();
    assert!(score == 100);

    let pred = rf.predict_votes(df.r(100..200).to_table(), None);

    let score = pred.cols[0]
        .array
        .inner::<IntegerArray<u64>>()
        .iter()
        .zip(pred.cols[1].array.inner::<IntegerArray<u64>>().iter())
        .zip(y_ins[100..200].iter())
        .map(|((&x1, &x2), &y)| ((x2 as f64 / (x1 + x2) as f64 > 0.5) as u64 == y) as u64)
        .sum::<u64>();

    assert!(score == 100);
}

#[test]
fn rf_cls_check_0_1_4ft_mixed_dtypes() {
    let mut rng = StdRng::seed_from_u64(1);
    let x_int: Vec<i64> = (0..NROW)
        .map(|_| *[0i64, 2 ^ 60].choose(&mut rng).unwrap())
        .collect();

    let x_cat: Vec<u64> = sample_0_1(&mut rng, NROW)
        .into_iter()
        .map(|x| x as u64)
        .collect();
    let x_float: Vec<f64> = (0..NROW)
        .map(|_| *[0f64, (2 ^ 60) as f64].choose(&mut rng).unwrap())
        .collect();
    let x_bool: Vec<bool> = sample_0_1(&mut rng, NROW)
        .into_iter()
        .map(|x| x == 1)
        .collect();

    let y: Vec<u64> = x_cat.clone();

    let unique_values = vec![String::from("false"), String::from("true")];
    let cat_unique_values = vec![String::from("cat_false"), String::from("cat_true")];

    let df_vec = vec![
        new_arr_categorical64("x_cat64", x_cat, &cat_unique_values),
        new_arr_i64("x_int64", x_int),
        new_arr_f64("x_float64", x_float),
        new_arr_bool("x_bool", x_bool),
    ];
    let rf = RandomForest::fit(
        Table::new("table".into(), df_vec.into()),
        Array::from_categorical64(CategoricalArray::from_slices(&y, &unique_values)),
        100,
        1,
        false,
        true,
        false,
        1,
        None,
    );
    let imp_table = rf.importance(false);

    let imp = imp_table.cols[imp_table.col_name_index("importance").unwrap()]
        .array
        .inner::<FloatArray<f64>>();

    assert!(imp[0] > 0.2);
    assert!(imp[1] < 0.05);
    assert!(imp[2] < 0.05);
    assert!(imp[3] < 0.05);
}

#[test]
fn rf_reg_check_0_1_3ft() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1: Vec<f64> = (0..NROW).map(|x| x as f64).collect();
    let y: Vec<f64> = x1.iter().map(|&x| 2. * x).collect();

    let a1 = new_arr_f64("x1", x1);
    let a2 = new_arr_i64("x2", sample_0_1(&mut rng, NROW));
    let a3 = new_arr_i64("x3", sample_0_1(&mut rng, NROW));

    let df_vec = vec![a1, a2, a3];
    let rf = RandomForest::fit(
        Table::new("table".into(), df_vec.into()),
        Array::from_float64(FloatArray::from_slice(&y)),
        100,
        1,
        false,
        true,
        false,
        1,
        None,
    );
    let imp_table = rf.importance(false);

    let imp = imp_table.cols[imp_table.col_name_index("importance").unwrap()]
        .array
        .inner::<FloatArray<f64>>();

    assert!(imp[0] > 1000.);
    assert!(imp[1] < 100.);
    assert!(imp[2] < 100.);
}

#[test]
fn rf_reg_predict_linear() {
    let nrow = 500;
    let mut rng = StdRng::seed_from_u64(1);

    let x1: Vec<f64> = (0..nrow).map(|_| rng.random_range(0.0..10.0)).collect();
    let y_ins = x1.clone();
    let df_vec = vec![new_arr_f64("x1", x1)];
    let y = FloatArray::from_slice(&y_ins[..400]);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df.r(0..400).to_table(),
        Array::from_float64(y),
        1000,
        1,
        true,
        false,
        false,
        1,
        None,
    );

    let pred = rf.predict(df.r(400..500).to_table(), 1, None);
    assert!(matches!(
        pred,
        Array::NumericArray(NumericArray::Float64(_))
    ));

    let Array::NumericArray(NumericArray::Float64(arr)) = pred else {
        unreachable!()
    };

    let mae: f64 = arr
        .iter()
        .zip(y_ins[400..].iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f64>()
        / 100.;
    assert!(mae < 0.02);
}

#[test]
fn rf_reg_oob_linear() {
    let nrow = 500;
    let mut rng = StdRng::seed_from_u64(1);

    let x1: Vec<f64> = (0..nrow).map(|_| rng.random_range(0.0..10.0)).collect();
    let y_ins = x1.clone();
    let df_vec = vec![new_arr_f64("x1", x1)];
    let y = FloatArray::from_slice(&y_ins);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df,
        Array::from_float64(y),
        1000,
        1,
        false,
        false,
        true,
        1,
        None,
    );

    let oob_pred = rf.oob(1);

    assert!(matches!(
        oob_pred,
        Array::NumericArray(NumericArray::Float64(_))
    ));

    let Array::NumericArray(NumericArray::Float64(arr)) = oob_pred else {
        unreachable!()
    };

    let mae: f64 = arr
        .iter()
        .zip(y_ins.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f64>()
        / 100.;
    assert!(mae < 0.08);
}

#[test]
#[should_panic(
    expected = "internal error: entered unreachable code: Votes for regression are not supported"
)]
fn rf_reg_predict_votes_should_panic() {
    let nrow = 500;
    let mut rng = StdRng::seed_from_u64(1);

    let x1: Vec<f64> = (0..nrow).map(|_| rng.random_range(0.0..10.0)).collect();
    let y_ins = x1.clone();
    let df_vec = vec![new_arr_f64("x1", x1)];
    let y = FloatArray::from_slice(&y_ins);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df.r(..400).to_table(),
        Array::from_float64(y),
        1000,
        1,
        true,
        false,
        false,
        1,
        None,
    );
    rf.predict_votes(df.r(400..).to_table(), None);
}

#[test]
#[should_panic(
    expected = "internal error: entered unreachable code: Votes for regression are not supported"
)]
fn rf_reg_oob_votes_should_panic() {
    let nrow = 500;
    let mut rng = StdRng::seed_from_u64(1);

    let x1: Vec<f64> = (0..nrow).map(|_| rng.random_range(0.0..10.0)).collect();
    let y_ins = x1.clone();
    let df_vec = vec![new_arr_f64("x1", x1)];
    let y = FloatArray::from_slice(&y_ins);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df,
        Array::from_float64(y),
        1000,
        1,
        true,
        false,
        false,
        1,
        None,
    );
    rf.oob_votes();
}

#[test]
fn rf_cls_predict_0_1_interactions_serialize_round_trip() {
    let nrow = 200;
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, nrow);
    let x2 = sample_0_1(&mut rng, nrow);

    let y_ins: Vec<_> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u64)
        .collect();

    let mut df_vec = vec![new_arr_i64("x1", x1), new_arr_i64("x2", x2)];
    for i in 1..10 {
        df_vec.push(new_arr_i64(
            &format!("rand{}", i),
            sample_0_1(&mut rng, nrow),
        ));
    }
    let unique_values = vec![String::from("false"), String::from("true")];
    let y = CategoricalArray::from_slices(&y_ins[0..100], &unique_values);
    let df = Table::new("table".into(), df_vec.into());

    let rf = RandomForest::fit(
        df.r(0..100).to_table(),
        Array::from_categorical64(y),
        1000,
        3,
        true,
        false,
        false,
        1,
        None,
    );

    let pred = rf.predict_votes_raw(df.r(100..200).to_table(), None);
    let rf_bytes = rf.to_bytes().unwrap();
    let deserialized_forest = RandomForest::from_bytes(&rf_bytes).unwrap();
    let pred_deserialized_forest =
        deserialized_forest.predict_votes_raw(df.r(100..200).to_table(), None);
    pred.predictions()
        .zip(pred_deserialized_forest.predictions())
        .for_each(|(x, y)| assert_eq!(x.0, y.0));
}
