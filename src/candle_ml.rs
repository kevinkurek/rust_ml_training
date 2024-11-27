use polars::prelude::*;
use candle_core::{Tensor, Device, Result as CandleResult};
use candle_nn::VarBuilder;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Read the CSV file into a Polars DataFrame
    let file = File::open("data.csv")?;
    let reader = BufReader::new(file);
    let df = CsvReader::new(reader)
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    println!("DataFrame: \n{}", df);

    // Step 2: Split the DataFrame into features and target
    let num_columns = df.width();
    if num_columns < 2 {
        return Err("Data must have at least two columns: features and a target.".into());
    }

    // Convert all but the last column into features
    let features: Vec<Vec<f32>> = (0..num_columns - 1)
        .map(|col_idx| {
            df.select_at_idx(col_idx)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .flatten()
                .map(|v| v as f32)
                .collect()
        })
        .collect();

    // Transpose features to row-major format
    let features: Vec<Vec<f32>> = (0..features[0].len())
        .map(|row_idx| features.iter().map(|col| col[row_idx]).collect())
        .collect();

    // Convert the last column into the target
    let target: Vec<f32> = df
        .select_at_idx(num_columns - 1)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|v| v as f32)
        .collect();

    println!("Features: {:?}\nTarget: {:?}", features, target);

    // Step 3: Convert data into Candle Tensors
    let device = Device::Cpu;

    let features_tensor = Tensor::from_vec(
        features.concat(),
        (features.len(), features[0].len()), // (num_samples, num_features)
        &device,
    )?;
    let target_tensor = Tensor::from_vec(target, (target.len(), 1), &device)?;

    // Step 4: Set up linear regression model
    let mut vb = VarBuilder::new(&device);
    let weights = vb.get_with_shape("weights", (features[0].len(), 1))?;
    let bias = vb.get_with_shape("bias", (1,))?;

    // Training parameters
    let learning_rate = 0.01;
    let num_epochs = 1000;

    // Step 5: Training loop
    for epoch in 0..num_epochs {
        // Forward pass: Predict
        let predictions = features_tensor.matmul(&weights)?.add(&bias)?;

        // Calculate mean squared error loss
        let error = predictions.sub(&target_tensor)?;
        let loss = error.squared()?.mean(0)?;

        // Backpropagation: Compute gradients
        let gradients = loss.backward()?;

        // Update weights and bias using gradients
        weights.update(&gradients, -learning_rate)?;
        bias.update(&gradients, -learning_rate)?;

        // Log training progress
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.to_vec::<f32>()?[0]);
        }
    }

    // Step 6: Make predictions
    let predictions = features_tensor.matmul(&weights)?.add(&bias)?;
    println!("Predictions: {:?}", predictions.to_vec::<f32>()?);

    Ok(())
}