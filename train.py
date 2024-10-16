import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from util import gradient_magnitude, weight_update_magnitude

# Defining loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error
    
def train_client(train_loader, global_model, lr, batch_size, num_local_epochs, model_type, device, previous_weights):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    input_dim = next(iter(train_loader))[0].shape[2]  # 6
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    avg_gradient_magnitude = 0.0
    avg_weight_update_magnitude = 0.0
    
    # Start training loop
    for epoch in range(0, num_local_epochs):
        h = model.init_hidden(batch_size)
        total_loss = 0.0
        for x, label in train_loader:
            if model_type == "GRU":
                h = h.data
            # Unpcak both h_0 and c_0
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])
            
            model.zero_grad()  # Set the gradients to zero before starting to do backpropragation
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())

            # Perform backpropragation
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (1/len(train_loader))
            # avg_loss += loss.item() * x.size(0)
            # print('x')
            # Compute gradients and weight updates
            gradient_magnitude_val = gradient_magnitude(model)
            weight_update_magnitude_val = weight_update_magnitude(model, previous_weights)
            avg_gradient_magnitude += gradient_magnitude_val * (1 / len(train_loader))
            avg_weight_update_magnitude += weight_update_magnitude_val * (1 / len(train_loader))

        # avg_loss = avg_loss / len(train_loader.dataset)
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, total_loss))
    # print(f'Avg Gradient Magnitude: {avg_gradient_magnitude:.4f},Avg Weight Update Magnitude: {avg_weight_update_magnitude:.4f}')

    return model, total_loss, avg_gradient_magnitude, avg_weight_update_magnitude

def evaluate(model, test_loader, label_sc, model_type, device, previous_weights):
    model.eval()
    model.to(device)
    all_out= []
    all_targ = []
    total_loss = 0.0
    avg_weight_update_magnitude = 0.0
    for k, test_data in enumerate(test_loader):
        outputs = []
        targets = []
        for batch_x, batch_y in test_data:
            inputs = batch_x.to(device)
            labels = batch_y.to(device)
            # Move each tensor in the tuple to the same device as the model            
            # h = model.init_hidden(inputs.shape[0])
            if model_type == "LSTM":
                h = tuple(h_item.to(device) for h_item in model.init_hidden(inputs.shape[0]))
            if model_type == "GRU":
                h = model.init_hidden(inputs.shape[0]).to(device)

            with torch.no_grad():
                out, h = model(inputs.to(device).float(), h)

            outputs.append(out.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
            # Calculate the loss
            loss = criterion(out, labels)
            # total_loss += loss.item()
            total_loss += loss.item() * 1/len(test_data)
            
            # Compute gradients and weight updates
            weight_update_magnitude_val = weight_update_magnitude(model, previous_weights)
            avg_weight_update_magnitude += weight_update_magnitude_val * (1 / len(test_data))

        concatenated_outputs = np.concatenate(outputs)
        concatenated_targets = np.concatenate(targets)
        all_out.append(label_sc[k].inverse_transform(concatenated_outputs).reshape(-1))
        all_targ.append(label_sc[k].inverse_transform(concatenated_targets).reshape(-1))

    extra_conc_out= np.concatenate(all_out)
    extra_conc_targ= np.concatenate(all_targ)
    # Calculate and print other metrics
    smape = calculate_smape(extra_conc_out, extra_conc_targ)
    mae = mean_absolute_error(extra_conc_targ, extra_conc_out)
    rmse = np.sqrt(mean_squared_error(extra_conc_targ, extra_conc_out))
    print(f"calc smape: {smape}%")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Calculate and print the average loss
    average_loss = total_loss / len(test_loader)
    print(f"Average Loss: {average_loss: }")
    # print(f'test Weight Update Magnitude: {avg_weight_update_magnitude:.4f}')
    return all_out, all_targ, average_loss, smape, mae, rmse, avg_weight_update_magnitude

def running_model_sum(current, next):
    if current == None:
        current = next
    else:
        for key in current:
            current[key] = current[key] + next[key]
    return current

def scale_model_state(model_state, scale):
    scaled_state = {key: value * scale for key, value in model_state.items()}
    return scaled_state
    
def calculate_smape(forecasted, actual):
    # Check for equal length of forecasted and actual arrays
    if len(forecasted) != len(actual):
        raise ValueError("Forecasted and actual arrays must have the same length.")
    total_smape = 0.0
    
    for i in range(len(forecasted)):
        Ft = forecasted[i]
        At = actual[i]
        numerator = abs(Ft - At)
        denominator = abs(Ft) + abs(At)
        
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        smape_i = (numerator / (denominator + epsilon)) * 100.0
        total_smape += smape_i
    
    # Calculate the average SMAPE over all data points
    average_smape = total_smape / len(forecasted)
    
    return average_smape
  
def poc_train_clients(train_loader, global_model, lr, batch_size, num_local_epochs, model_type, device):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()
    input_dim = next(iter(train_loader))[0].shape[2]  # 6
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Start training loop
    for epoch in range(0, num_local_epochs):
        h = model.init_hidden(batch_size)
        total_loss = 0.0
        for x, label in train_loader:
            if model_type == "GRU":
                h = h.data
            # Unpcak both h_0 and c_0
            elif model_type == "LSTM":
                h = tuple([e.data for e in h])
            
            model.zero_grad()  # Set the gradients to zero before starting to do backpropragation
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())

            # Perform backpropragation
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (1/len(train_loader))
            # avg_loss += loss.item() * x.size(0)
        # avg_loss = avg_loss / len(train_loader.dataset)
        print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, num_local_epochs, total_loss))

    return model, total_loss

def poc_evaluate(model, test_loader, label_sc, model_type, device):
    model.eval()
    model.to(device)
    all_out= []
    all_targ = []
    total_loss = 0.0
    for k, test_data in enumerate(test_loader):
        outputs = []
        targets = []
        for batch_x, batch_y in test_data:
            inputs = batch_x.to(device)
            labels = batch_y.to(device)
            # Move each tensor in the tuple to the same device as the model            
            # h = model.init_hidden(inputs.shape[0])
            if model_type == "LSTM":
                h = tuple(h_item.to(device) for h_item in model.init_hidden(inputs.shape[0]))
            if model_type == "GRU":
                h = model.init_hidden(inputs.shape[0]).to(device)

            with torch.no_grad():
                out, h = model(inputs.to(device).float(), h)

            outputs.append(out.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
            # Calculate the loss
            loss = criterion(out, labels)
            # total_loss += loss.item()
            total_loss += loss.item() * 1/len(test_data)
            
        concatenated_outputs = np.concatenate(outputs)
        concatenated_targets = np.concatenate(targets)
        all_out.append(label_sc[k].inverse_transform(concatenated_outputs).reshape(-1))
        all_targ.append(label_sc[k].inverse_transform(concatenated_targets).reshape(-1))

    extra_conc_out= np.concatenate(all_out)
    extra_conc_targ= np.concatenate(all_targ)
    # Calculate and print other metrics
    smape = calculate_smape(extra_conc_out, extra_conc_targ)
    mae = mean_absolute_error(extra_conc_targ, extra_conc_out)
    rmse = np.sqrt(mean_squared_error(extra_conc_targ, extra_conc_out))
    print(f"calc smape: {smape}%")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Calculate and print the average loss
    average_loss = total_loss / len(test_loader)
    print(f"Average Loss: {average_loss: }")
    # print(f'test Weight Update Magnitude: {avg_weight_update_magnitude:.4f}')
    return all_out, all_targ, average_loss, smape, mae, rmse