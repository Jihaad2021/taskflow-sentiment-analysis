"""Generate test CSV files for testing."""

import pandas as pd
from pathlib import Path

# Ensure directory exists
Path("test_data").mkdir(exist_ok=True)

# Test 1: Small dataset (already have sample_reviews.csv)

# Test 2: Medium dataset with edge cases
medium_data = {
    'comment': [
        "Great product! Love it.",
        "Terrible quality, very disappointed.",
        "",  # Empty
        "Good value for money",
        "NOT RECOMMENDED!!!",  # Caps + special chars
        "It's okay, nothing special",
        "Best purchase ever! 😊",  # Emoji
        "Worst product I've ever bought 😡",
        "Meh... could be better",
        "Amazing! Highly recommend to everyone!",
    ] * 5  # 50 rows
}

df_medium = pd.DataFrame(medium_data)
df_medium.to_csv('test_data/medium_test.csv', index=False)
print("✅ Created: test_data/medium_test.csv (50 rows)")

# Test 3: Dataset with missing values
missing_data = {
    'review_id': range(1, 21),
    'comment': [
        "Good product" if i % 3 != 0 else None
        for i in range(1, 21)
    ],
    'rating': [5, 4, None, 3, 2, 1, 5, None, 4, 3] * 2
}

df_missing = pd.DataFrame(missing_data)
df_missing.to_csv('test_data/with_missing.csv', index=False)
print("✅ Created: test_data/with_missing.csv (20 rows with NaN)")

# Test 4: Long text
long_data = {
    'feedback': [
        "This is a very long review. " * 50,  # Very long
        "Short",
        "Medium length review with some details about the product.",
    ] * 5  # 15 rows
}

df_long = pd.DataFrame(long_data)
df_long.to_csv('test_data/long_text.csv', index=False)
print("✅ Created: test_data/long_text.csv (15 rows with long text)")

print("\n✅ All test data created!")