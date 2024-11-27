use polars::prelude::*;
use std::fs::File;
use std::io::BufReader;
use linfa::Dataset;
use linfa_linear::LinearRegression;
use linfa::traits::{Fit, Predict};
use linfa::metrics::SingleTargetRegression;
use ndarray::{Array1, Array2};
use std::error::Error;
use anyhow::Result;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {

    // Create a large CSV file
    // create_large_csv("data.csv")?;

    // Read the CSV file into a DataFrame
    let file = File::open("data.csv")?;
    let reader = BufReader::new(file);
    let df = CsvReader::new(reader)
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    // Print the first 5 rows of the DataFrame
    println!("{:?}", df.head(Some(5)));

    // run example from linfa docs
    // example_linfa_docs();

    // train linear regression model
    let target = "Column5";
    train_linear_regression(&df, &target)?;

    Ok(())
}

fn example_linfa_docs () {
    // Load the diabetes dataset
    let dataset = linfa_datasets::diabetes();
    // println!("{:?}", dataset);

    // Split the dataset into training and testing sets (80% train, 20% test)
    let (train, test) = dataset.split_with_ratio(0.8);

    // println!("Train: {:?}", train);
    // println!("Test: {:?}", test);

    // Fit the linear regression model on the training data
    let model = LinearRegression::new().fit(&train).expect("Failed to fit model");

    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());

    // Predict the target values for the test set
    let predictions = model.predict(&test);
    // println!("Predictions: {:?}", predictions);

    // Calculate R² score using the test set
    let r2 = SingleTargetRegression::r2(&test, &predictions);
    println!("R² from test set: {:?}", r2.unwrap());
}

fn create_large_csv(file_path: &str) -> Result<()> {
    // Create a DataFrame with 5 columns and 100,000 rows
    let mut df = DataFrame::new(vec![
        Series::new("Column1", (1..=100_000).collect::<Vec<_>>()),
        Series::new("Column2", (1..=100_000).map(|x| x * 2).collect::<Vec<_>>()),
        Series::new("Column3", (1..=100_000).map(|x| x * 3).collect::<Vec<_>>()),
        Series::new("Column4", (1..=100_000).map(|x| x * 4).collect::<Vec<_>>()),
        Series::new("Column5", (1..=100_000).map(|x| x * 5).collect::<Vec<_>>()),
    ])?;

    // Write the DataFrame to a CSV file
    let mut file = File::create(file_path)?;
    CsvWriter::new(&mut file)
        .has_header(true)
        .finish(&mut df)?;

    Ok(())
}

// train linear regression model
fn train_linear_regression(df: &DataFrame, target: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let target_column = target;
    let feature_columns: Vec<&str> = df
        .get_column_names()
        .iter()
        .filter(|&&col| col != target_column)
        .cloned()
        .collect();

    // Step 3: Extract targets as Array1<f64>
    let targets_series = df.column(target_column)?.cast(&DataType::Float64)?;
    let targets = targets_series
        .f64()?
        .into_no_null_iter()
        .collect::<Array1<f64>>();

    // Step 4: Extract features as Array2<f64>
    let num_samples = df.height();
    let num_features = feature_columns.len();
    let mut records = Array2::<f64>::zeros((num_samples, num_features));

    for (i, &col_name) in feature_columns.iter().enumerate() {
        let col_series = df.column(col_name)?.cast(&DataType::Float64)?;
        let col_data = col_series
            .f64()?
            .into_no_null_iter()
            .collect::<Array1<f64>>();

        // Assign the column data to the records array
        records
            .column_mut(i)
            .assign(&col_data);
    }

    // Step 5: Create a Linfa Dataset
    let dataset = Dataset::new(records, targets);

    // Step 6: Split the dataset into training and testing sets
    let (train, test) = dataset.split_with_ratio(0.8);

    // Step 7: Fit a linear regression model
    let model = LinearRegression::new().fit(&train)?;

    // Step 8: Make predictions on the test set
    let predictions = model.predict(&test);

    // Step 9: Calculate the R² score
    let r2 = predictions.r2(&test).unwrap_or_else(|_| {
        eprintln!("Warning: R² score could not be calculated.");
        0.0
    });

    println!("R² score on the test set: {}", r2);

    Ok(())
}