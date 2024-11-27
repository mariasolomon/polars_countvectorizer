use polars::prelude::*;

fn main() -> Result<(), PolarsError> {
    // Create the DataFrame
    let mut df = df![
        "doc1" => [ "This is the first document.", "Another document." ],
        "doc2" => [ "This is the second one.", "Another document." ]
    ]?;

    println!("Original DataFrame:");
    println!("{:?}", df);
    Ok(())
}