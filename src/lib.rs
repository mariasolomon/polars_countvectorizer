use linfa_preprocessing::CountVectorizer;
use ndarray::Array1;
use polars::prelude::*;
use rayon::prelude::*;
use sprs::CsMat;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use pyo3::exceptions::PyRuntimeError;

/// Calculates cosine distances and adds a column to the DataFrame
#[pyfunction]
fn process_cosine_distances_py(py_df: PyDataFrame) -> PyResult<PySeries> {

    let df: DataFrame = py_df.into();

    let doc1: Vec<String> = df.column("doc1")
                        .map_err(|e| PyRuntimeError::new_err(format!("Polars error : {:?}", e)))? 
                        .str()
                        .map_err(|e| PyRuntimeError::new_err(format!("Polars error: {:?}", e)))? 
                        .into_no_null_iter()
                        .map(|s| s.to_string())
                        .collect();
    
    let doc2: Vec<String> = df.column("doc2")
                        .map_err(|e| PyRuntimeError::new_err(format!("Polars error: {:?}", e)))? 
                        .str()
                        .map_err(|e| PyRuntimeError::new_err(format!("Polars error: {:?}", e)))? 
                        .into_no_null_iter()
                        .map(|s| s.to_string())
                        .collect();


    let cosine_distances: Vec<f32> = doc1
        .into_iter()
        .zip(doc2.into_iter())
        .par_bridge()
        .map(|(d1, d2)| {
            let documents_array = Array1::from(vec![d1, d2]);
            calculate_cosine_distance(&documents_array)
        })
        .collect();

    let cosine_series = Series::new("cosine_distance".into(), cosine_distances);

    Ok(PySeries(cosine_series))

}

fn calculate_cosine_distance(documents_array: &Array1<String>) -> f32 {
    let vectorizer = CountVectorizer::params().normalize(true);
    let model = vectorizer.fit(documents_array).unwrap();

    let transformed_matrix: CsMat<usize> = model.transform(documents_array);
    let dense_matrix = transformed_matrix.to_dense();
    let dense_matrix_f32 = dense_matrix.map(|x| *x as f32);

    let a = dense_matrix_f32.row(0).to_owned();
    let b = dense_matrix_f32.row(1).to_owned();

    let dot_product = a.dot(&b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    let cosine_similarity = dot_product / (norm_a * norm_b);

    format!("{:.2}", cosine_similarity).parse().unwrap()
}

#[pymodule]
fn polars_countvectorizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_cosine_distances_py, m)?)?;
    Ok(())
}
