use polars::prelude::*;
use candle_core::{Result as CandleResult, DType, Device, Tensor};
// use candle_nn::VarBuilder;
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
    assert!(num_columns > 1, "DataFrame must have at least 2 columns");

    // Convert all but the last column into features
    let features: Vec<Vec<f64>> = (0..num_columns - 1)
        .map(|col_idx| {
            df.select_at_idx(col_idx)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .flatten()
                .collect()
                // .map(|v| v as f32)
        })
        .collect();

    // Transpose features to row-major format
    let features: Vec<Vec<f64>> = (0..features[0].len())
        .map(|row_idx| features.iter().map(|col| col[row_idx]).collect())
        .collect();

    // Convert the last column into the target
    let target: Vec<f64> = df
        .select_at_idx(num_columns - 1)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
        // .map(|v| v as f32)

    println!("Features: {:?}\nTarget: {:?}", features, target);

    // Step 3: Convert data into Candle Tensors
    let device = Device::Cpu;

    let features_tensor = Tensor::from_vec(
        features.concat(),
        (features.len(), features[0].len()), // (num_samples, num_features)
        &device,
    )?;
    let target_tensor = Tensor::from_vec(target, (target.len(), 1), &device)?;

    // Step 4: Initialize model
    let vs = VarStore::new(Device::Cpu);
    let mut vb = VarBuilder::from(&vs);
    let linear = Linear::new(features[0].len(), 1, &mut vb)?;

    // Training parameters
    let learning_rate = 0.01;
    let num_epochs = 10;

    // Step 5: Training loop
    for epoch in 0..num_epochs {
        // Forward pass: Predict
        let predictions = features_tensor.matmul(&weights)?.add(&bias)?;

        // Calculate mean squared error loss
        let error = predictions.sub(&target_tensor)?;
        let loss = error.powf(2.0)?.mean(0)?;
        println!("Loss: {:?}", loss.to_vec1::<f32>()?);

        // Backpropagation: Compute gradients
        loss.backward()?;

        // Update weights and bias using gradients
        if let Some(weights_grad) = weights.gradient()? {
            weights = weights.sub(&weights_grad.mul_scalar(learning_rate)?)?;
        }

        if let Some(bias_grad) = bias.gradient()? {
            bias = bias.sub(&bias_grad.mul_scalar(learning_rate)?)?;
        }

        // Log training progress
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.to_vec0::<f32>()?[0]);
        }
    }

    // Step 6: Make predictions
    let predictions = features_tensor.matmul(&weights)?.add(&bias)?;
    println!("Predictions: {:?}", predictions.to_vec0::<f32>()?);

    Ok(())
}