import numpy as np

""" Random coreset selection """
def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    idx = np.random.choice(x_train.shape[0], coreset_size, False)
    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx,:])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train    

""" K-center coreset selection """
def k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    dists = np.full(x_train.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train, current_id)
    idx = [ current_id ]

    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)

    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx,:])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train

def update_distance(dists, x_train, current_id):
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists

""" Uncertainty-based coreset selection """
def uncertainty_based(model, task_id, x_coreset, y_coreset, x_train, y_train, coreset_size):
    uncertainties = get_uncertainty_per_dataset(model, task_id, x_train)[0].sum(axis=-1)  # Epistemic
    sorted_idx = uncertainties.argsort()[::-1]
    idx = sorted_idx[:coreset_size]

    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx,:])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train

def get_uncertainty_per_dataset(model, task_id, inputs):
    T = model.no_pred_samples

    temp_batch_size = 256
    N = inputs.shape[0]
    if temp_batch_size > N:
        temp_batch_size = N
    total_batch = int(np.ceil(N * 1.0 / temp_batch_size))
    batch_predictions = []

    # Loop over all batches
    for i in range(total_batch):
        start_ind = i*temp_batch_size
        end_ind = np.min([(i+1)*temp_batch_size, N])
        batch = inputs[start_ind:end_ind, :]
        pred_i = model.prediction_prob(batch, task_id)
        batch_predictions.append(pred_i)
    
    predictions = np.concatenate(batch_predictions, axis=1)

    epistemics = []
    aleatorics = []

    # loop over each sample in the inputs
    for sample in range(predictions.shape[1]):
        p_hat = predictions[:, sample, :]
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        epistemic = np.diag(epistemic)
        epistemics.append(epistemic)

        aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
        aleatoric = np.diag(aleatoric)
        aleatorics.append(aleatoric)

    epistemic = np.vstack(epistemics)  # (batch_size, categories)
    aleatoric = np.vstack(aleatorics)  # (batch_size, categories)

    return epistemic, aleatoric
        

