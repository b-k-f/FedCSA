import numpy as np
from train import train_client, poc_train_clients, running_model_sum, scale_model_state, evaluate, poc_evaluate
from util import objective_f, client_selection, save_weights
from ad_method import detect_anomalies_with_autoencoder
import torch
import torch.nn as nn

def fedavg(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []
    weight_lst = []
    # cl_grad_mag = [0.0] * n_clients
    # cl_upd_mag = [0.0] * n_clients
    # loss_lst = [0.0] * n_clients
    # ae_lst = [0.0] * n_clients
    
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    stability_threshold = 10
    tmp_cl = clients
    n_tmp_cl = len(tmp_cl)
    copy_test_loader = test_loader
    absence_count = {i: 0 for i in range(n_clients)}

    for t in range(max_rounds):    
        cl_grad_mag = []
        cl_upd_mag = []
        loss_lst = []
        ae_lst = []
    
        weight_avg = None
        print("\nstarting round {}".format(t))
        # clients = list(range(0, n_clients))
        print("clients: ", clients)
        global_model.eval() # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device) #run on gpu
        previous_global_weights = save_weights(global_model)

        for k,cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k,n_clients-1, cid))
            if t ==0:
                previous_weights = previous_global_weights
            else:
                previous_weights = save_weights(weight_lst[k])
            
            local_model, train_loss, train_gradient_magnitude, train_weight_update_magnitude = train_client(
                train_loader = client_train_loader[k],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device, 
                previous_weights = previous_weights)
            
            cl_upd_mag.append(train_gradient_magnitude)
            cl_grad_mag.append(train_weight_update_magnitude)
            loss_lst.append(train_loss)
            weight_lst.append(local_model)
            
            is_anomaly, reconstruction_error = detect_anomalies_with_autoencoder(global_model, threshold=0.005)
            print(f"loss error: {reconstruction_error:.4f}")
            ae_lst.append(reconstruction_error)
            
            print(tmp_cl)
            if k in tmp_cl:
                print('llll')
                weight_avg = running_model_sum(weight_avg, local_model.state_dict())
                
        weight_avg = scale_model_state(weight_avg, 1/(n_tmp_cl))
        # set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse, test_weight_update_magnitude = evaluate(global_model, test_loader, label_sc, model_type, device, previous_global_weights)       
        
        # print('global gradient magnitude: ', gradient_magnitude(global_model))
        # print('global weight update magnitude: ', gradient_magnitude(global_model))
        
        #save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)

        scores = objective_f(loss_lst, cl_grad_mag, cl_upd_mag, ae_lst)
        print(scores)
        
        tmp_cl = client_selection(scores, 0.75)
        n_tmp_cl = len(tmp_cl)
        print(tmp_cl, n_tmp_cl)

        # Update absence count for each client
        current_set = set(tmp_cl)
        for i, client in enumerate(clients):
            print(i, client)
            if i not in current_set:
                absence_count[client] += 1
            else:
                absence_count[client] = 0
            if absence_count[client] >= stability_threshold:
                del absence_count[client]
                print('deleted client ' , client)
        # Remove clients whose absence count exceeds the threshold
        clients = [client for client in clients if client in absence_count]
        n_clients = len(clients)
        print(absence_count)
        print("Updated clients: ", clients)

        test_loader = [copy_test_loader[i] for i in tmp_cl] 
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def poc(global_model, client_train_loader, test_loader, label_sc, n_clients, num_clients_per_round, batch_size, num_local_epochs, lr, max_rounds, model_type, device, powd, ratio):
    
    all_smape = []  # list of all rounds smape
    all_loss = []   # list of all rounds loss     
    all_mae = []
    all_rmse = []
    copy_test_loader = test_loader
    
    for t in range(max_rounds):
        print("\nStarting  round {}".format(t))
        global_model.eval()  # turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc. 
        global_model = global_model.to(device)  # run on gpu
        
        # Step 1: Sample candidate set A of d clients without replacement
        # d = min(50, n_clients)  # Using a candidate set size d=50 as an example
        candidate_clients = np.random.choice(n_clients, powd, replace=False)
        candidate_losses = []
        print("Clients: ", candidate_clients)
        
        # Step 2: Estimate local losses for sampled clients
        for cid in candidate_clients:
            local_model, local_loss = poc_train_clients(
                train_loader=client_train_loader[cid],
                global_model=global_model,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=1,  # Using 1 epoch for quick loss estimation
                model_type=model_type,
                device=device)
            candidate_losses.append((cid, local_loss))
        
        # Step 3: Select top m clients with highest local losses
        candidate_losses.sort(key=lambda x: x[1], reverse=True)
        selected_clients = [cid for cid, _ in candidate_losses[:num_clients_per_round]]
        
        weight_avg = None
        for k, cid in enumerate(selected_clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1, num_clients_per_round, cid))
            
            local_model, _ = poc_train_clients(
                train_loader=client_train_loader[cid],
                global_model=global_model,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device)
            
            # Aggregate local model weights
            weight_avg = running_model_sum(weight_avg, local_model.state_dict())
        
        weight_avg = scale_model_state(weight_avg, 1/num_clients_per_round)
        
        # Set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse = poc_evaluate(global_model, test_loader, label_sc, model_type, device)
        
        # Save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        test_loader = [copy_test_loader[i] for i in selected_clients] 
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

def selective(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device, percentile):
    
    all_smape = []  # List of all rounds' SMAPE
    all_loss = []   # List of all rounds' loss     
    all_mae = []
    all_rmse = []
    copy_test_loader = test_loader
    
    clients = np.arange(0, n_clients) # choose client ids w/out repetition
    
    for t in range(max_rounds):
        weight_avg = None
        print("\nStarting avg round {}".format(t))
        print("Clients: ", clients)
        global_model.eval()  # Turn off during model evaluation: Dropouts Layers, BatchNorm Layers, etc.
        global_model = global_model.to(device)  # Run on GPU
        
        local_models = []
        losses = []
        credits = []
        Mt = 0  # Sum of reciprocal of the loss values
        
        for k, cid in enumerate(clients):
            print("Round {}, starting client {}/{}, ID: {}".format(t, k+1, n_clients, cid))
        
            local_model, train_loss = poc_train_clients(
                train_loader=client_train_loader[cid],
                global_model=global_model,
                lr=lr,
                batch_size=batch_size,
                num_local_epochs=num_local_epochs,
                model_type=model_type,
                device=device)
            
            local_models.append(local_model)
            losses.append(train_loss)
            Mt += 1 / train_loss

        N = sum(len(loader.dataset) for loader in client_train_loader)  # Total number of data points
        
        # Compute the threshold based on the percentile
        threshold = np.percentile(losses, percentile)
        print(f"Computed threshold based on {percentile}th percentile: {threshold}")
        
        for i, model in enumerate(local_models):
            loss_k_t = losses[i]
            n_k = len(client_train_loader[i].dataset)  # Number of training data points on edge device i
            if loss_k_t > threshold:
                c_k_t = 0
            else:
                c_k_t = (n_k / N) * (1 / loss_k_t) / Mt
            credits.append(c_k_t)
         # Normalize credits
        credit_sum = sum(credits)
        normalized_credits = [c_k_t / credit_sum for c_k_t in credits]
        print(normalized_credits)

        valid_clients = [(j, local_models[j], normalized_credits[j]) for j in range(len(clients)) if normalized_credits[j] > 0]
        valid_clients_indices = [j for j in range(len(clients)) if normalized_credits[j] > 0]

        for k, local_model, credit_score in valid_clients:
            # Multiply the credit score by the local model weights
            weighted_state_dict = {key: value * credit_score for key, value in local_model.state_dict().items()}
            # Do the average of local model weights with each client in a single round
            weight_avg = running_model_sum(weight_avg, weighted_state_dict)
            
        # weight_avg = scale_model_state(weight_avg, 1 / len(valid_clients))

        # Set global model parameters for the next step
        global_model.load_state_dict(weight_avg)
        # test_loader = [copy_test_loader[i] for i in valid_clients_indices]
        print(valid_clients_indices)
        outputs, target, loss, smape, mae, rmse = poc_evaluate(global_model, test_loader, label_sc, model_type, device)       
        
        # Save final round accuracy and loss in a list to return 
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse

class Ae(nn.Module):
    def __init__(self, input_dim):
        super(Ae, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def calculate_anomaly_score(autoencoder, data, device):
    autoencoder = autoencoder.to(device)
    data = data.to(device)
    reconstructions = autoencoder(data)
    reconstruction_error = torch.mean((data - reconstructions) ** 2, dim=1)
    return reconstruction_error

def calculate_credit_scores(anomaly_scores, n_clients, L):
    inv_anomaly_scores = anomaly_scores ** -L
    credit_scores = n_clients * inv_anomaly_scores / torch.sum(n_clients * inv_anomaly_scores)
    return credit_scores

def aggregate_weights(client_weights, credit_scores):
    global_weight = sum(credit_score * client_weight for credit_score, client_weight in zip(credit_scores, client_weights))
    return global_weight

def abnormal(global_model, client_train_loader, test_loader, label_sc, n_clients, batch_size, num_local_epochs, lr, max_rounds, model_type, device):
    
    all_smape = [] #list of all rounds smape
    all_loss=[] #list of all rounds loss     
    all_mae = []
    all_rmse = []

    # Initialize the anomaly detection model
    input_dim = sum(p.numel() for p in global_model.parameters())  # Total number of parameters in the model
    autoencoder = Ae(input_dim)
    L = 1.0  # Hyperparameter for credit score calculation

    for t in range(max_rounds):
        weight_avg = None
        print("\nstarting avg round {}".format(t))
        clients = np.arange(0, n_clients) # choose client ids w/out repetition
        print("clients: ", clients)
        global_model.eval()
        global_model = global_model.to(device)
        
        client_weights = []
        anomaly_scores = []
        n_client_data = []

        for k, cid in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, k+1, n_clients, cid))
            
            local_model, _ = poc_train_clients(
                train_loader = client_train_loader[cid],
                global_model = global_model,
                lr = lr,
                batch_size = batch_size,
                num_local_epochs = num_local_epochs,
                model_type = model_type,
                device = device)
            
            client_weights.append(local_model.state_dict())
            n_client_data.append(len(client_train_loader[cid].dataset))

            # Flatten model parameters to calculate anomaly scores
            flattened_weights = torch.cat([param.view(-1) for param in local_model.parameters()])
            anomaly_score = torch.mean(calculate_anomaly_score(autoencoder, flattened_weights.unsqueeze(0),device))
            anomaly_scores.append(anomaly_score.item())
        
        anomaly_scores = torch.tensor(anomaly_scores)
        n_client_data = torch.tensor(n_client_data, dtype=torch.float32)
        credit_scores = calculate_credit_scores(anomaly_scores, n_client_data, L)
        
        # weight_avg = aggregate_weights(client_weights, credit_scores)
        for i in range(len(clients)):
            # Multiply the credit score by the local model weights
            weighted_state_dict = {key: value * credit_score[i] for key, value in local_model[i].state_dict().items()}
            # Do the average of local model weights with each client in a single round
            weight_avg = running_model_sum(weight_avg, weighted_state_dict)
            
        global_model.load_state_dict(weight_avg)
        outputs, target, loss, smape, mae, rmse = poc_evaluate(global_model, test_loader, label_sc, model_type, device)
        
        all_loss.append(loss)
        all_smape.append(smape)
        all_mae.append(mae)
        all_rmse.append(rmse)
        
    return outputs, target, all_loss, all_smape, all_mae, all_rmse