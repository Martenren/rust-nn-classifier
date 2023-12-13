extern crate piston_window;

use egui::*;


mod utils;
mod neural_network;

use arrow::record_batch::RecordBatch;
use std::path::Path;


fn main() {
    let iris_file_path: &Path = Path::new("../data/iris.csv");
    let df_iris: RecordBatch = utils::read_csv(iris_file_path);
    let neural_network = neural_network::NeuralNetwork::new(vec![2, 3, 2]);
    let mut accuracy: f64 = 0.0;
    let mut total: i32 = 0;

    for row in 0..df_iris.num_rows() {
        let sepal_length = df_iris.column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(row);

        let sepal_width = df_iris.column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(row);

        let class_name = df_iris.column(4)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
            .value(row);

        let outputs = neural_network.calculate_outputs(vec![sepal_length, sepal_width]);

        let class: i32 = neural_network.classify(&outputs).unwrap() as i32;

        let class_index = match class_name {
            "Setosa" => 0,
            "Versicolor" => 1,
            "Virginica" => 2,
            _ => -1,
        };

        println!("predicted_class: {:?}", class);
        println!("actual_class: {:?}", class_index);

        accuracy += if class == class_index { 1.0 } else { 0.0 };
        total += 1;
    }
    accuracy /= total as f64;
    println!("accuracy: {:?}", accuracy);
}
