use minarrow::{Array, IntegerArray};
use minrf::RandomForestClassifier;
use rand::{Rng, SeedableRng, rngs::StdRng};
const NROW: usize = 100;

fn sample_0_1(rng: &mut impl Rng, k: usize) -> Vec<i64> {
    (0..k)
        .map(|_| rng.random_range(0..=1))
        .collect::<Vec<i64>>()
}

fn new_arr_i64(x: Vec<i64>) -> Array {
    Array::from_int64(IntegerArray::<i64>::from_slice(&x))
}

#[test]
fn rf_check_0_1_3ft() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, NROW);
    let y: Vec<u32> = x1.iter().map(|&x| x as u32).collect();

    let a1 = new_arr_i64(x1);
    let a2 = new_arr_i64(sample_0_1(&mut rng, NROW));
    let a3 = new_arr_i64(sample_0_1(&mut rng, NROW));

    let yy = IntegerArray::<u32>::from_slice(&y);

    let df_vec = vec![a1, a2, a3];
    let rf = RandomForestClassifier::fit(df_vec, yy, 2, NROW, 3, 100, 1, false, true, false, 1);
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

    let y_ins: Vec<u32> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u32)
        .collect();
    let y = IntegerArray::<u32>::from_slice(&y_ins);

    let mut df_vec = vec![new_arr_i64(x1), new_arr_i64(x2)];
    for _i in 1..100 {
        df_vec.push(new_arr_i64(sample_0_1(&mut rng, NROW)));
    }
    let ncol = df_vec.len();

    let rf = RandomForestClassifier::fit(df_vec, y, 2, NROW, ncol, 1000, 10, false, true, false, 1);
    let imp = rf.importance_normalised();

    assert!(imp[0] > 0.5);
    assert!(imp[1] > 0.5);
    for i in 2..imp.len() {
        // println!("{:?}", i);
        assert!(imp[i] < 0.2);
    }
}

#[test]
fn rf_oob_0_1_interactions() {
    let mut rng = StdRng::seed_from_u64(1);
    let x1 = sample_0_1(&mut rng, NROW);
    let x2 = sample_0_1(&mut rng, NROW);

    let y_ins: Vec<u32> = x1
        .iter()
        .zip(x2.iter())
        .map(|row| (*row.0 == 1 && *row.1 == 1) as u32)
        .collect();
    let y = IntegerArray::<u32>::from_slice(&y_ins);

    let mut df_vec = vec![new_arr_i64(x1), new_arr_i64(x2)];
    for _i in 1..10 {
        df_vec.push(new_arr_i64(sample_0_1(&mut rng, NROW)));
    }
    let ncol = df_vec.len();
    let yy = y.clone();

    let rf = RandomForestClassifier::fit(df_vec, y, 2, NROW, ncol, 1000, 3, false, false, true, 1);

    let score = rf
        .oob()
        .iter()
        .zip(yy.iter())
        .map(|(&x, &y)| (x == y) as u64)
        .sum::<u64>();
    assert!(score == 100);
}
