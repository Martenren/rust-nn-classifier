use std::fs::File;
use std::path::Path;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{Schema, DataType, Field};
use std::sync::Arc;
use arrow::array::{RecordBatch};


pub(crate) fn read_csv(
    csv_file_path: &Path,
) -> RecordBatch {
    let schema : Schema = Schema::new(vec![
        // Field::new("id", DataType::Int64, false),
        Field::new("sepal.length", DataType::Float64, false),
        Field::new("sepal.width", DataType::Float64, false),
        Field::new("petal.length", DataType::Float64, false),
        Field::new("petal.width", DataType::Float64, false),
        Field::new("variety", DataType::Utf8, false),
    ]);

    let file = File::open(csv_file_path).unwrap();

    let mut csv = ReaderBuilder::new(Arc::new(schema)).has_header(true).build(file).unwrap();

    let batch = csv.next().unwrap().unwrap();

    return batch;
}

pub(crate) struct Datapoint {
    pub(crate) inputs: Vec<f64>,
    pub(crate) expected_output: f64
}