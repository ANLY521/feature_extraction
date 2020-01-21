#!/usr/bin/env python
import argparse
import numpy as np
from util import load_function_words, load_reviews, split_data

# do not alter this function
def check_splits(train, test, X, ids):
    """verify that all data is retained after splitting"""
    # check X
    train_X = train[0]
    val_X = test[0]
    sum_after = train_X.sum()+val_X.sum()
    assert sum_after == X.sum(), \
        f"Sum of features in Train+Test {sum_after} must equal sum of features before splitting {X.sum}"

    all_ids = train[1] + test[1]
    assert set(all_ids) == set(ids), "Set of ids in Train+Test must equal set of ids before splitting"
    # if we didn't crash, everything's good!
    print("Split checks passed!")



def main(data_file, vocab_path):
    """extract function word features from a text file"""

    # load resources and text file
    function_words = load_function_words(vocab_path)

    reviews, ids = load_reviews(data_file)


    # TODO: appropriately shape and fill this matrix
    review_features = np.zeros((1,1), dtype=np.int)
    # row is which review
    # column is which word

    print(f"Numpy array has shape {review_features.shape} and dtype {review_features.dtype}")

    # TODO: Calculate these from review_features
    most_common_count = 0
    most_common_word = ""
    print(f"Most common word: {most_common_word}, count: {most_common_count}")

    # TODO: Find any features that weren't in the data (i.e. columns that sum to 0)
    zero_inds = []
    if len(zero_inds)>0:
        print("No instances found for: ")
        for ind in zero_inds:
            print(f"  {function_words[ind]}")
    else:
        print("All function words found")


    matrix_sum = review_features.sum()
    print(f"Sum of raw count matrix: {matrix_sum}")

    # TODO: make a binary feature vector from your count vector
    word_binary = np.copy(review_features)
    word_binary_sum = word_binary.sum()
    print(f"Sum of binary matrix: {word_binary_sum}")

    # TODO: normalize features for review length (divide rows by number of function words in the review)
    # HINT: each row should sum to 1
    norm_reviews = np.copy(review_features)
    norm_reviews_sum = norm_reviews.sum()
    print(f"Sum of normed matrix: {norm_reviews_sum}")

    # TODO: remove features from <review_features> that occur less than <min_count> times
    min_count = 100
    min_matrix = np.copy(review_features)
    min_matrix_shape = min_matrix.shape
    print(f"Shape after removing features that occur < {min_count} times: {min_matrix_shape}")

    #TODO: split the dataset by updating the function above

    train, val = split_data(review_features, ids, 0.3)

    # Code below that all your data has been retained in your splits; do not edit.
    # Must all print True

    check_splits(train, val, review_features, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_practice.txt",
                        help='path to input with one review per line')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
