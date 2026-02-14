from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, abs, length, avg
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
import numpy as np

# ----------------------------------
# Create Spark Session
# ----------------------------------
spark = SparkSession.builder \
    .appName("BigDataAssignmentSpark") \
    .getOrCreate()

# ----------------------------------
# Load Dataset
# ----------------------------------
books_df = spark.read.text("books/")

print("Total lines:", books_df.count())

# ----------------------------------
# Metadata Extraction
# ----------------------------------
books_df = books_df.withColumn(
    "title", regexp_extract("value", r"Title:\s*(.*)", 1)
).withColumn(
    "author", regexp_extract("value", r"Author:\s*(.*)", 1)
).withColumn(
    "release_date", regexp_extract("value", r"Release Date:\s*(.*)", 1)
).withColumn(
    "language", regexp_extract("value", r"Language:\s*(.*)", 1)
)

books_df.select("title", "author", "release_date", "language").show(10, truncate=False)

# ----------------------------------
# Year Extraction & Analysis
# ----------------------------------
books_df = books_df.withColumn(
    "year", regexp_extract("release_date", r"(\d{4})", 1)
)

print("Books per year:")
books_df.filter(books_df.year != "") \
    .groupBy("year").count().orderBy("year").show()

print("Language distribution:")
books_df.filter(books_df.language != "") \
    .groupBy("language").count().orderBy("count", ascending=False).show()

print("Average title length:")
books_df.filter(books_df.title != "") \
    .select(avg(length("title"))).show()

# ----------------------------------
# TF-IDF Processing
# ----------------------------------
tokenizer = Tokenizer(inputCol="value", outputCol="words")
words_df = tokenizer.transform(books_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
clean_df = remover.transform(words_df)

tf = HashingTF(inputCol="filtered_words", outputCol="tf_features", numFeatures=10000)
tf_df = tf.transform(clean_df)

idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

tfidf_df.select("tfidf_features").show(5, truncate=False)

# ----------------------------------
# Cosine Similarity Example
# ----------------------------------
v1 = tfidf_df.select("tfidf_features").first()[0].toArray()
v2 = tfidf_df.select("tfidf_features").take(2)[1][0].toArray()

if np.linalg.norm(v1) != 0 and np.linalg.norm(v2) != 0:
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
else:
    similarity = 0

print("Cosine Similarity between first two documents:", similarity)

# ----------------------------------
# Author Influence Network
# ----------------------------------
edges = books_df.alias("a").join(
    books_df.alias("b"),
    (abs(col("a.year").cast("int") - col("b.year").cast("int")) <= 5) &
    (col("a.author") != col("b.author")) &
    (col("a.year") != "")
).select(
    col("a.author").alias("author1"),
    col("b.author").alias("author2")
)

edges.show(10, truncate=False)

print("Out-degree (influence given):")
edges.groupBy("author1").count().orderBy("count", ascending=False).show()

print("In-degree (influence received):")
edges.groupBy("author2").count().orderBy("count", ascending=False).show()
books_df.write.mode("overwrite").csv("output_metadata")
edges.write.mode("overwrite").csv("output_author_network")


# ----------------------------------
spark.stop()
