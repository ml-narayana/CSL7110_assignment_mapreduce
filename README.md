# Big Data Processing with Hadoop MapReduce & Apache Spark

This repository contains the complete implementation and analysis for
Assignment 1 -- MapReduce and Apache Spark. The project demonstrates
distributed batch processing using Hadoop and large-scale in-memory
analytics using Spark on real-world textual data.

------------------------------------------------------------------------

## Repository Structure

BigData-Hadoop-Spark-Assignment/ │ ├── Hadoop/WordCount.java ├──
Spark/spark_analysis.py ├── books/ ├── screenshots/ └── README.md

------------------------------------------------------------------------

## Technologies Used

-   Apache Hadoop 3.x
-   Apache Spark 4.x
-   Java
-   Python (PySpark)
-   HDFS

------------------------------------------------------------------------

## Hadoop WordCount

Custom Java MapReduce implementation to compute word frequency.

Run:

hadoop jar WordCount.jar WordCount /input /output

------------------------------------------------------------------------

## Spark Analytics

-   Metadata extraction
-   TF-IDF similarity
-   Author influence network

Run:

spark-submit Spark/spark_analysis.py

------------------------------------------------------------------------

## Dataset

Project Gutenberg public domain books.

------------------------------------------------------------------------

## Results

-   Successful distributed processing
-   Large-scale text analytics
-   Graph-based influence modeling

------------------------------------------------------------------------

## Author

Ashwini

------------------------------------------------------------------------

CSL7110 -- Big Data Systems
