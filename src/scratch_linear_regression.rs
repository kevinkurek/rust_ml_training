use nalgebra::{DMatrix, DVector};

struct LinearRegression {
    weights: DVector<f64>,
    bias: f64,
}

impl LinearRegression {
    fn new() -> Self {
        LinearRegression {
            weights: DVector::zeros(1),
            bias: 0.0,
        }
    }

    fn predict(&self, x: &DMatrix<f64>) -> DVector<f64> {
        x * &self.weights + DVector::from_element(x.nrows(), self.bias)
    }

    fn train(&mut self, x: &DMatrix<f64>, y: &DVector<f64>, epochs: usize, lr: f64) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let errors = &predictions - y;
            let gradient = x.transpose() * &errors / (x.nrows() as f64);
            self.weights -= lr * gradient;
            self.bias -= lr * errors.mean();
        }
    }
}

fn generate_data(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let x = DMatrix::from_fn(n, 1, |_, _| rand::random::<f64>() * 10.0);
    let noise = DVector::from_fn(n, |_, _| rand::random::<f64>() - 0.5);
    let y = &x * 2.0 + noise.add_scalar(1.0);
    (x, y)
}

fn main() {
    let (x, y) = generate_data(10);
    println!("x: {}", x);
    println!("y: {}", y);

    let mut model = LinearRegression::new();
    model.train(&x, &y, 1000, 0.01);

    println!("Weights: {:?}", model.weights);
    println!("Bias: {:?}", model.bias);

    let test_data = DMatrix::from_row_slice(1, 1, &[5.0]);
    let prediction = model.predict(&test_data);
    println!("Prediction for input 5.0: {:?}", prediction);
}