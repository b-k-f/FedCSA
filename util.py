import numpy as np
import torch

np.random.seed(0)

def client_selection(values, threshold):
    # Step 1: Identify the maximum value
    max_value = max(values)
    # Step 2: Define the threshold value
    threshold_value = max_value * threshold
    # Step 3: Collect indices of values that are within the threshold of the maximum value
    indices = [index for index, value in enumerate(values) if value >= threshold_value]
    return indices

def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def objective_f(loss, grad_magnitude, weight_upd_magnitude, autoenc):
    norm_gradient_magnitudes = normalize(np.array(grad_magnitude))
    norm_weight_update_magnitudes = normalize(np.array(weight_upd_magnitude))
    norm_loss = normalize(np.array(loss))
    norm_ae = normalize(np.array(autoenc))
    
    # Invert the normalized values of the magnitudes to make them minimizing objectives
    inv_norm_gradient_magnitudes = 1 - norm_gradient_magnitudes
    inv_norm_weight_update_magnitudes = 1 - norm_weight_update_magnitudes
    inv_norm_loss = 1 - norm_loss
    inv_norm_ea = 1 - norm_ae
    # Combine metrics into a single score for each client without accuracy
    alpha, gamma, delta = 0.34,0.33,0.33  # Ensure weights sum to 1
    raw_scores = (
        alpha * inv_norm_gradient_magnitudes +
        # beta * inv_norm_weight_update_magnitudes +
        gamma * inv_norm_loss + 
        delta * inv_norm_ea
    )
    # Normalize the raw scores to sum to 1
    total_score = np.sum(raw_scores)
    proportionate_scores = raw_scores / total_score

    # Use accuracy for anomaly handling Penalize clients with low accuracy
    # accuracy_threshold = 0.7  # Example threshold
    # if accuracy < accuracy_threshold:
    #     proportionate_scores[i] *= 0.1  # Penalize by reducing the score

    # Re-normalize the scores after penalization
    # total_score = np.sum(proportionate_scores)
    # proportionate_scores = proportionate_scores / total_score
    # print('proportionate_scores: ', proportionate_scores)
    return proportionate_scores

def gradient_magnitude(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def weight_update_magnitude(model, previous_weights):
    update_magnitude = 0.0
    for p, prev_p in zip(model.parameters(), previous_weights):
        update_magnitude += torch.sum((p.data - prev_p.data) ** 2).item()
    return update_magnitude

def save_weights(model):
    return [p.data.clone() for p in model.parameters()]

def move_sliding_window(data, window_size, inputs_cols_indices, label_col_index):
    """
    data: numpy array including data
    window_size: size of window
    inputs_cols_indices: col indices to include
    """

    # (# instances created by movement, seq_len (timestamps), # features (input_len))
    inputs = np.zeros((len(data) - window_size, window_size, len(inputs_cols_indices)))
    labels = np.zeros(len(data) - window_size)

    for i in range(window_size, len(data)):
        inputs[i - window_size] = data[i - window_size : i, inputs_cols_indices]
        labels[i - window_size] = data[i, label_col_index]
    inputs = inputs.reshape(-1, window_size, len(inputs_cols_indices))
    labels = labels.reshape(-1, 1)
    print(inputs.shape, labels.shape)

    return inputs, labels

def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
