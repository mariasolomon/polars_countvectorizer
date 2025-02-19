# polars_countvectorizer

polars_countvectorizer is a Rust-based Python library that provides text vectorization and cosine distance calculation, optimized for use with Polars DataFrames. The library offers a function to compute the cosine distance between two text columns in a Polars DataFrame, making it useful for text similarity and document analysis tasks.

## Features

- Cosine Distance Calculation: Computes the cosine similarity between two text columns in a Polars DataFrame.
- CountVectorizer Integration: Uses the CountVectorizer from the linfa_preprocessing crate to vectorize text data.
- Optimized for Performance: Leverages parallel processing via rayon for faster computations.


# Usage
```
import polars as pl
import polars_countvectorizer

df = pl.DataFrame({
    "doc1": ["apple orange banana", "dog cat mouse", "car bike bus"],
    "doc2": ["apple banana", "cat dog", "bus car bike"]
})

result = df.select(
    polars_countvectorizer.process_cosine_distances_py("doc1", "doc2")
)

print(result)
```
