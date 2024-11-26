use linfa_preprocessing::CountVectorizer;
use ndarray::Array1;
use polars::prelude::*;
use rayon::prelude::*;
use sprs::CsMat;


fn process_cosine_distances(df: &mut DataFrame) -> Result<DataFrame, PolarsError> {
    let doc1: Vec<String> = df.column("doc1")?.str()?.into_no_null_iter().map(|s| s.to_string()).collect();
    let doc2: Vec<String> = df.column("doc2")?.str()?.into_no_null_iter().map(|s| s.to_string()).collect();

    // Calculate cosine distances in parallel
    let cosine_distances: Vec<f32> = doc1
        .into_iter()
        .zip(doc2.into_iter())
        .par_bridge()
        .map(|(d1, d2)| {
            let documents_array = Array1::from(vec![d1, d2]);
            calculate_cosine_distance(&documents_array)
        })
        .collect();

    // Create a new column for cosine distances
    let cosine_series = Series::new("cosine_distance".into(), cosine_distances);

    // Add the cosine distance column to the DataFrame
    let result_df = df.with_column(cosine_series)?;

    Ok(result_df.clone())
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
    let cosine_similarity = (dot_product / (norm_a * norm_b) ) * 100.0;

    format!("{:.2}", cosine_similarity).parse().unwrap()
}


fn main() -> Result<(), PolarsError> {
    // Create the DataFrame
    let mut df = df![
        "doc1" => [ "This is the first document.", "Another document." ],
        "doc2" => [ "This is the second one.", "Another document." ]
    ]?;

    // Display the original DataFrame
    println!("Original DataFrame:");
    println!("{:?}", df);

    // Process the DataFrame to calculate cosine distances
    let result_df = process_cosine_distances(&mut df)?;

    // Display the resulting DataFrame with cosine distances
    println!("DataFrame with Cosine Distance:");
    println!("{:?}", result_df);

    Ok(())
}