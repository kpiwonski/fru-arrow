use minarrow::{Array, IntegerArray};
use xrf::{Forest, RfControl};

mod attribute;

mod classification;
use classification::DataFrame as DataFrameClassification;

pub struct RandomForestClassifier(Forest<DataFrameClassification>);

impl RandomForestClassifier {
    fn importance(&self) -> Vec<f64> {
        self.0.importance().map(|(_, x)| x).collect()
    }

    fn importance_normalised(&self) -> Vec<f64> {
        self.0.importance_normalised().map(|(_, x)| x).collect()
    }

    fn oob(&self) -> Vec<u32> {
        self.0.oob_predictions().map(|(_, x)| x).collect()
    }

    fn fit(
        x: Vec<Array>,
        y: IntegerArray<u32>,
        ncat: u32,
        nrow: usize,
        ncol: usize,
        control: RfControl,
    ) -> Self {
        let df = DataFrameClassification::new(x, y, ncat, ncol, nrow);
        let forest = RandomForestClassifier(Forest::new(&df, &control));
        forest
    }

    fn predict(&self, x: Vec<Array>, ncat: u32, nrow: usize, ncol: usize) -> Vec<u32> {
        let df = DataFrameClassification::new(x, IntegerArray::from_slice(&[]), ncat, ncol, nrow);
        let pred = self.0.predict(&df);
        pred.predictions().map(|(_, x)| x).collect()
    }
}
