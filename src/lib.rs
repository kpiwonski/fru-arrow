use minarrow::{Array, IntegerArray};
use xrf::{Forest, RfRng};

mod attribute;

mod classification;
use classification::DataFrame as DataFrameClassification;

pub struct RandomForestClassifier(Forest<DataFrameClassification>);

impl RandomForestClassifier {
    pub fn importance(&self) -> Vec<f64> {
        // TODO check if has importance
        let mut imp: Vec<_> = self.0.importance().collect();
        imp.sort_unstable_by_key(|(k, _)| *k);
        imp.into_iter().map(|(_, x)| x).collect()
    }

    pub fn importance_normalised(&self) -> Vec<f64> {
        let mut imp: Vec<_> = self.0.importance_normalised().collect();
        imp.sort_unstable_by_key(|(k, _)| *k);
        imp.into_iter().map(|(_, x)| x).collect()
    }

    pub fn oob(&self) -> Vec<u32> {
        let mut oob: Vec<_> = self.0.oob().map(|(e, v)| (e, v.collapse())).collect(); // TODO change to rng

        oob.sort_unstable_by_key(|(k, _)| *k);
        oob.into_iter().map(|(_, x)| x).collect()
    }

    pub fn oob_votes(&self) -> Vec<Vec<u32>> {
        let mut oob: Vec<_> = self.0.oob().map(|(e, v)| (e, v.0.clone())).collect();

        oob.sort_unstable_by_key(|(k, _)| *k);
        oob.into_iter().map(|(_, x)| x).collect()
    }

    pub fn fit(
        x: Vec<Array>,
        y: IntegerArray<u32>,
        ncat: u32,
        nrow: usize,
        ncol: usize,
        trees: usize,
        tries: usize,
        save_forest: bool,
        importance: bool,
        oob: bool,
        seed: u64,
    ) -> Self {
        let df = DataFrameClassification::new(x, y, ncat, ncol, nrow);
        let forest = RandomForestClassifier(Forest::new(
            &df,
            trees,
            tries,
            save_forest,
            importance,
            oob,
            seed,
            // 1, //TODO
        ));
        forest
    }

    pub fn predict(
        &self,
        x: Vec<Array>,
        ncat: u32,
        nrow: usize,
        ncol: usize,
        threads: usize,
        seed: u64,
    ) -> Vec<u32> {
        // let mut rng = RfRng::from_seed(seed, 1); //TODO correct sometime
        let df = DataFrameClassification::new(x, IntegerArray::default(), ncat, ncol, nrow);
        let pred = self.0.predict_parallel(&df, threads);
        let mut pred: Vec<_> = pred.predictions().map(|(e, v)| (e, v.collapse())).collect();

        pred.sort_unstable_by_key(|(k, _)| *k);
        pred.into_iter().map(|(_, x)| x).collect()
    }
}
