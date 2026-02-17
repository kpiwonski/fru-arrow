use minarrow::CategoricalArray;
use minarrow::{Array, BooleanArray, FieldArray, FloatArray, IntegerArray, Table};
use minrf::RandomForestClassifier;
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
fn rf_check_0_1_3ft() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, NROW);
    let y: Vec<u64> = x1.iter().map(|&x| x as u64).collect();

    let a1 = new_arr_i64("x1", x1);
    let a2 = new_arr_i64("x2", sample_0_1(&mut rng, NROW));
    let a3 = new_arr_i64("x3", sample_0_1(&mut rng, NROW));

    let unique_values = vec![String::from("false"), String::from("true")];

    let df_vec = vec![a1, a2, a3];
    let rf = RandomForestClassifier::fit(
        Table::new("table".into(), df_vec.into()),
        CategoricalArray::from_slices(&y, &unique_values),
        100,
        1,
        false,
        true,
        false,
        1,
        None,
    );
    let imp = rf.importance();

    assert!(imp[0] > 0.2);
    assert!(imp[1] < 0.01);
    assert!(imp[2] < 0.01);
}

#[test]
fn rf_importance_0_1_interactions() {
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

    let rf = RandomForestClassifier::fit(
        Table::new("table".into(), df_vec.into()),
        CategoricalArray::from_slices(&y_ins, &unique_values),
        1000,
        10,
        false,
        true,
        false,
        1,
        None,
    );
    let imp = rf.importance_normalised();

    assert!(imp[0] > 0.5);
    assert!(imp[1] > 0.5);
    for i in 2..imp.len() {
        assert!(imp[i] < 0.2);
    }
}

#[test]
fn rf_oob_0_1_interactions() {
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

    let rf = RandomForestClassifier::fit(
        Table::new("table".into(), df_vec.into()),
        y,
        1000,
        3,
        false,
        false,
        true,
        1,
        None,
    );

    let score = rf
        .oob()
        .iter()
        .zip(yy.iter())
        .map(|(&x, &y)| (x == y) as u64)
        .sum::<u64>();
    assert!(score == 100);

    let score = rf
        .oob_votes()
        .iter()
        .zip(yy.iter())
        .map(|(x, &y)| ((x[1] as f64 / (x[0] + x[1]) as f64 > 0.5) as u64 == y) as u64)
        .sum::<u64>();
    assert!(score == 100);
}

#[test]
fn rf_predict_0_1_interactions() {
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

    let rf = RandomForestClassifier::fit(
        df.slice(0, 100).to_table(),
        y,
        1000,
        3,
        true,
        false,
        false,
        1,
        None,
    );

    let score = rf
        .predict(df.slice(100, 100).to_table(), 1, None)
        .iter()
        .zip(y_ins[100..200].iter())
        .map(|(&x, &y)| (x == y) as u64)
        .sum::<u64>();
    assert!(score == 100);

    let score = rf
        .predict_votes(df.slice(100, 100).to_table(), None)
        .iter()
        .zip(y_ins[100..200].iter())
        .map(|(x, &y)| ((x[1] as f64 / (x[0] + x[1]) as f64 > 0.5) as u64 == y) as u64)
        .sum::<u64>();
    assert!(score == 100);
}

#[test]
fn rf_check_0_1_4ft_mixed_dtypes() {
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
    let rf = RandomForestClassifier::fit(
        Table::new("table".into(), df_vec.into()),
        CategoricalArray::from_slices(&y, &unique_values),
        100,
        1,
        false,
        true,
        false,
        1,
        None,
    );
    let imp = rf.importance();

    assert!(imp[0] > 0.2);
    assert!(imp[1] < 0.05);
    assert!(imp[2] < 0.05);
    assert!(imp[3] < 0.05);
}
