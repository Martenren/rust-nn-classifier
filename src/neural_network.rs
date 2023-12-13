use ndarray::Data;
use crate::utils::Datapoint;

struct Layer {
    num_inputs: usize,
    num_outputs: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

fn activation_function( x: f64) -> f64 {
    let mut res:f64 = 0.;
    if x>0.{
        res = 1.0 / (1.0 + (-x).exp())
    } else {
        res = (x).exp() / (1.0 + (x).exp())
    }
    return res
}

impl Layer {
    fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let mut weights: Vec<Vec<f64>> = Vec::new();

        for i in 0..num_outputs{
            weights.push(Vec::new());
            for _j in 0..num_inputs{
                weights[i].push(1.0);
            }
        }

        let mut biases: Vec<f64> = Vec::new();
        for _ in 0..num_outputs{
                biases.push(1.0);
        }

        Layer{
            num_inputs,
            num_outputs,
            weights,
            biases,
        }

    }

    fn calculate_outputs(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activation_values: Vec<f64> = Vec::new();

        //println!("weights: {:?}", self.weights);

        //println!("num inputs: {:?}", self.num_inputs);
        //println!("num outputs: {:?}", self.num_outputs);

        for node_out in 0..self.num_outputs {
            let mut weighted_input = 0.0;
            for node_in in 0..self.num_inputs {
                weighted_input += inputs[node_in] * self.weights[node_out][node_in];
            }
            weighted_input += self.biases[node_out];
            activation_values.push(activation_function(weighted_input));
        }
        return activation_values;
    }

    pub(crate) fn neuron_cost(&self, output_activation: f64, expected_output: f64) -> f64 {
        let error: f64 = output_activation - expected_output;
        return error * error
    }
}


pub(crate) struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork{
    pub(crate) fn new(layer_size: Vec<i32>) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..layer_size.len()-1 {
            let layer = Layer::new(layer_size[i] as usize, layer_size[i+1] as usize);
            layers.push(layer);
        }

        NeuralNetwork {
            layers
        }
    }

    pub(crate) fn calculate_outputs(&self, mut inputs: Vec<f64>) -> Vec<f64> {
        for layer in &self.layers{
            inputs = layer.calculate_outputs(inputs); //becomes input for next layer
        }

        return inputs; //output
    }

    pub(crate) fn classify(&self, outputs: &[f64]) -> Result<usize, &'static str> {
        if let Some((index, &_max_value)) = outputs
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
        {
            let class = index;
            Ok(class)
        } else {
            Err("No element matches the minimum value")
        }
    }

    fn cost(&self, datapoint: Datapoint) -> f64{
        let mut outputs: Vec<f64> = self.calculate_outputs(datapoint.inputs);
        let output_layer: &Layer = &self.layers[self.layers.len() - 1];
        let mut cost: f64 = 0.;

        for neuronOut in 0..outputs.len(){
            cost += output_layer.neuron_cost(outputs[neuronOut], datapoint.expected_output)
        }

        return cost;
    }
    fn total_cost(&self, data:Vec<Datapoint>) -> f64{
        let mut total_cost: f64 = 0.;

        for datapoint in data{
            self.cost(datapoint);
        }

        return total_cost
    }
}