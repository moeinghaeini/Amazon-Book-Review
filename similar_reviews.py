from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SimilarReviews") \
    .getOrCreate()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords and special characters"""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return float(intersection) / union if union > 0 else 0.0

def main():
    # Read the sampled reviews data
    reviews_df = spark.read.csv("sampled_data/Books_rating_sampled.csv", header=True, inferSchema=True)
    
    # Select only the review text and create an RDD
    reviews_rdd = reviews_df.select("Id", "review/text").rdd
    
    # Preprocess reviews and create (id, tokens) pairs
    processed_reviews = reviews_rdd.map(lambda x: (x[0], set(preprocess_text(x[1]))))
    
    # Create pairs of reviews for comparison
    review_pairs = processed_reviews.cartesian(processed_reviews) \
        .filter(lambda x: x[0][0] < x[1][0])  # Avoid duplicate pairs and self-comparisons
    
    # Calculate Jaccard similarity for each pair
    similar_pairs = review_pairs.map(lambda x: (
        (x[0][0], x[1][0]),  # Review IDs
        jaccard_similarity(x[0][1], x[1][1])  # Jaccard similarity
    ))
    
    # Filter pairs with similarity above threshold
    SIMILARITY_THRESHOLD = 0.5
    similar_pairs = similar_pairs.filter(lambda x: x[1] >= SIMILARITY_THRESHOLD)
    
    # Sort by similarity score in descending order
    similar_pairs = similar_pairs.sortBy(lambda x: x[1], ascending=False)
    
    # Take top 10 similar pairs
    top_similar_pairs = similar_pairs.take(10)
    
    # Print results
    print("\nTop 10 Similar Review Pairs:")
    print("=" * 50)
    for (id1, id2), similarity in top_similar_pairs:
        print(f"Review IDs: {id1} - {id2}")
        print(f"Jaccard Similarity: {similarity:.3f}")
        print("-" * 50)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 