use minarrow::{Array, CategoricalArray, IntegerArray, Table, Vec64};
use xrf::{Forest, RfRng};

mod attribute;

mod classification;
use classification::DataFrame as DataFrameClassification;

use crate::classification::DecisionBasicType;

pub struct RandomForestClassifier {
    forest: Forest<DataFrameClassification>,
    decision_unique_values: Vec64<String>,
}

impl RandomForestClassifier {
    pub fn importance(&self) -> Vec<f64> {
        // TODO check if has importance
        let mut imp: Vec<_> = self.forest.importance().collect();
        imp.sort_unstable_by_key(|(k, _)| *k);
        imp.into_iter().map(|(_, x)| x).collect()
    }

    pub fn importance_normalised(&self) -> Vec<f64> {
        let mut imp: Vec<_> = self.forest.importance_normalised().collect();
        imp.sort_unstable_by_key(|(k, _)| *k);
        imp.into_iter().map(|(_, x)| x).collect()
    }

    pub fn oob(&self) -> CategoricalArray<DecisionBasicType> {
        let mut rng = RfRng::from_seed(1, 1);
        let mut oob: Vec<_> = self
            .forest
            .oob()
            .map(|(e, v)| (e, v.collapse_empty_random(&mut rng)))
            .collect();

        oob.sort_unstable_by_key(|(k, _)| *k);
        let res: Vec<_> = oob.into_iter().map(|(_, x)| x).collect();
        CategoricalArray::from_slices(&res, &self.decision_unique_values)
    }

    pub fn oob_votes(&self) -> Vec<Vec<u64>> {
        let mut oob: Vec<_> = self.forest.oob().map(|(e, v)| (e, v.0.clone())).collect();

        oob.sort_unstable_by_key(|(k, _)| *k);
        oob.into_iter().map(|(_, x)| x).collect()
    }

    pub fn fit(
        x: Table,
        y: CategoricalArray<DecisionBasicType>,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
    ) -> Self {
        let decision_unique_values = y.unique_values.clone();
        let df = DataFrameClassification::new(x, y);
        let forest = Forest::new(
            &df,
            trees,
            tries,
            save_forest,
            importance,
            oob,
            seed,
            // 1, //TODO
        );
        RandomForestClassifier {
            forest,
            decision_unique_values,
        }
    }

    pub fn predict(
        &self,
        x: Table,
        threads: usize,
        seed: u64,
    ) -> CategoricalArray<DecisionBasicType> {
        let mut rng = RfRng::from_seed(seed, 1);
        let df = DataFrameClassification::new(x, CategoricalArray::default());
        let pred = self.forest.predict_parallel(&df, threads);
        let mut pred: Vec<_> = pred
            .predictions()
            .map(|(e, v)| (e, v.collapse_empty_random(&mut rng)))
            .collect();

        pred.sort_unstable_by_key(|(k, _)| *k);
        let res: Vec<_> = pred.into_iter().map(|(_, x)| x).collect();
        CategoricalArray::from_slices(&res, &self.decision_unique_values)
    }
}
