# import numpy as np

# def topsis(decision_matrix, weights, impacts):
#     """
#     Perform TOPSIS method.
    
#     Parameters:
#     decision_matrix (2D list or numpy array): Decision matrix where rows are alternatives and columns are criteria.
#     weights (list): Weights for each criterion.
#     impacts (list): '+' for beneficial criteria, '-' for non-beneficial criteria.
    
#     Returns:
#     list: Ranking of alternatives based on TOPSIS.
#     """
#     decision_matrix = np.array(decision_matrix, dtype=float)
#     norm_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
#     weighted_matrix = norm_matrix * weights
#     ideal_positive = np.max(weighted_matrix, axis=0) * (np.array(impacts) == '+') + \
#                      np.min(weighted_matrix, axis=0) * (np.array(impacts) == '-')
#     ideal_negative = np.min(weighted_matrix, axis=0) * (np.array(impacts) == '+') + \
#                      np.max(weighted_matrix, axis=0) * (np.array(impacts) == '-')
#     dist_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
#     dist_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))
#     scores = dist_negative / (dist_positive + dist_negative)
#     rankings = np.argsort(-scores) + 1
#     return rankings, scores

# # if __name__ == "__main__":
# #     # Example decision matrix
# #     decision_matrix = [
# #         [250, 16, 12, 5],
# #         [200, 16, 8, 3],
# #         [300, 32, 16, 4],
# #         [275, 8, 8, 4],
# #         [225, 16, 16, 2]
# #     ]
# #     weights = [0.25, 0.25, 0.25, 0.25]
# #     impacts = ['+', '+', '-', '+']

# #     rankings, scores = topsis(decision_matrix, weights, impacts)
# #     print("Rankings:", rankings)
# #     print("Scores:", scores)


import numpy as np
import pandas as pd

def topsis(decision_matrix, weights, impacts):
    # Convert inputs to numpy arrays for easier manipulation
    decision_matrix = np.array(decision_matrix)
    weights = np.array(weights)
    
    # Normalize the decision matrix
    norm_matrix = np.zeros_like(decision_matrix, dtype=float)
    for i in range(len(decision_matrix[0])):
        norm_matrix[:, i] = decision_matrix[:, i] / np.sqrt(np.sum(decision_matrix[:, i]**2))

    # Weighted normalized decision matrix
    weighted_matrix = norm_matrix * weights

    # Find the positive and negative ideal solutions
    ideal_positive = np.max(weighted_matrix, axis=0)
    ideal_negative = np.min(weighted_matrix, axis=0)

    # Calculate the Euclidean distances from the ideal solutions
    dist_positive = np.sqrt(np.sum((weighted_matrix - ideal_positive)**2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted_matrix - ideal_negative)**2, axis=1))

    # Calculate the Topsis score and rank
    scores = dist_negative / (dist_positive + dist_negative)
    rankings = np.argsort(scores)[::-1] + 1  # Rankings start from 1

    return rankings, scores
