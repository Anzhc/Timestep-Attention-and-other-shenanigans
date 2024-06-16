#Custom_train_functions
#Line 8 or so

import torch.nn.functional as F
from sklearn.cluster import KMeans
import warnings
...

#Just add all those at the end

def timestep_attention(run_number, loss_map, max_timesteps, b_size, device):
    #global adjusted_probabilities_print
    mean, std = 0.00, 1.00

    if loss_map:
        all_timesteps = torch.arange(1, max_timesteps, device=device)
        all_losses = torch.tensor([loss_map.get(t.item(), 1) for t in all_timesteps], device=device)

        # Adjust the losses based on the specified criteria
        adjusted_losses = adjust_losses(all_losses)

        # Calculate new probabilities with the adjusted losses
        adjusted_probabilities = adjusted_losses / adjusted_losses.sum()

        sampled_indices = torch.multinomial(adjusted_probabilities, b_size, replacement=True)
        skewed_timesteps = all_timesteps[sampled_indices]
    else:
        # Generate log-normal samples for timesteps as the fallback
        lognorm_samples = torch.distributions.LogNormal(mean, std).sample((b_size,)).to(device)
        normalized_samples = lognorm_samples / lognorm_samples.max()
        skewed_timesteps = (normalized_samples * (max_timesteps - 1)).long()

    # Log the adjusted probabilities
    #dir_name = f"H:\\TimestepAttention\\run{run_number}"
    #if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)
    # List existing files and find the next available file number
    #existing_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f)) and 'run_probabilities_' in f]
    #highest_num = 0
    #for file_name in existing_files:
    #    parts = file_name.replace('run_probabilities_', '').split('.')
    #    try:
    #        num = int(parts[0])
    #        if num > highest_num:
    #            highest_num = num
    #    except ValueError:
            # This handles the case where the file name does not end with a number
    #        continue

    # Determine the filename for the new log
    #new_file_num = highest_num + 1
    #file_out = os.path.join(dir_name, f"run_probabilities_{new_file_num}.txt")

    #adjusted_probabilities_print = adjusted_probabilities.cpu().tolist()
    #timesteps_probs_str = ', '.join(map(str, adjusted_probabilities_print))
    #with open(file_out, 'w') as file:
    #    file.write(timesteps_probs_str + '\n')

    return skewed_timesteps, skewed_timesteps

def update_loss_map_ema(current_loss_map, new_losses, timesteps, update_fraction=0.5):
    if not isinstance(new_losses, torch.Tensor):
        new_losses = torch.tensor(new_losses, dtype=torch.float32)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.tensor.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint*")
    timesteps = torch.tensor(timesteps, dtype=torch.long, device=new_losses.device)
    # Initialize all timesteps with a basic loss value of 0.1 if they are not already present
    for ts in range(1, 1000):  # Assuming timesteps from 1 to 999
        if ts not in current_loss_map:
            current_loss_map[ts] = 1

    aggregated_losses = new_losses.view(-1).tolist()

    # Define decay weights for adjusting adjacent timesteps
    decay_weights = [0.01, 0.03, 0.09, 0.25, 0.35, 0.50]

    for i, timestep_value in enumerate(timesteps.tolist()):
        loss_value = aggregated_losses[i] if i < len(aggregated_losses) else 0.0

        # Correctly handle the definition of current_average_loss
        # This ensures it's always defined before being used
        current_average_loss = current_loss_map.get(timestep_value, 1)

        # Directly update the timestep with the specified update fraction
        new_average_loss = current_average_loss + (loss_value - current_average_loss) * update_fraction
        current_loss_map[timestep_value] = new_average_loss

        # Apply decayed adjustments for adjacent timesteps
        for offset, weight in enumerate(decay_weights, start=1):
            for direction in [-1, 1]:
                adjacent_timestep = timestep_value + direction * offset
                if 1 <= adjacent_timestep < 1000:  # Ensure it's within the valid range
                    if adjacent_timestep in current_loss_map:
                        adjacent_current_loss = current_loss_map[adjacent_timestep]
                        # Calculate the decayed loss value based on distance and main loss value
                        decayed_loss_value = loss_value - (loss_value - current_average_loss) * weight
                        # Apply update fraction to move towards the decayed loss value
                        new_adjacent_loss = adjacent_current_loss + (decayed_loss_value - adjacent_current_loss) * update_fraction
                        current_loss_map[adjacent_timestep] = new_adjacent_loss

    return current_loss_map

def apply_loss_adjustment(loss, timesteps, loss_map, c_step, sched_train_steps):

    # Retrieve and calculate probabilities from the loss map
    all_timesteps = torch.arange(1, 1000, device=loss.device)  # Assuming max_timesteps is 1000
    all_losses = torch.tensor([loss_map.get(t.item(), 1.0) for t in all_timesteps], dtype=torch.float32, device=loss.device)
    
    # Calculate adjusted probabilities for each timestep
    adjusted_probabilities = all_losses / all_losses.sum()

    # Calculate the mean probability (average selection chance)
    mean_probability = adjusted_probabilities.mean()

    # Retrieve probabilities for specific sampled timesteps
    timestep_probabilities = adjusted_probabilities[timesteps - 1]

    # Calculate multipliers based on probabilities relative to the mean
    multipliers = timestep_probabilities / mean_probability

    schedule_start = 1
    schedule_move = -1
    loss_curve_scale = schedule_start + (schedule_move * (c_step/sched_train_steps))
    # Apply the 'loss_curve_scale' to modulate the multiplier effect
    # This reduces the effect of extreme multipliers, both high and low
    multipliers = 1 + (multipliers - 1) * loss_curve_scale

    # Adjust loss for each timestep based on its multiplier
    adjusted_loss = loss * multipliers.view(-1, 1, 1, 1)

    return adjusted_loss

def exponential_weighted_loss(noise_pred, target, alphas_cumprod, timesteps, loss_map, reduction="none", boundary_shift=0.0):
    # Compute the MAE loss
    mae_loss = F.l1_loss(noise_pred, target, reduction="none")

    # Calculate the mean of the MAE loss
    mean_mae_loss = mae_loss.mean()

    # Apply boundary shift if any
    mean_mae_loss += boundary_shift

    # Create a weighting map based on the exponential differences around the mean
    weight_map = torch.exp(mae_loss - mean_mae_loss)

    # Apply the weight map to the MAE loss
    weighted_loss = mae_loss * weight_map

    # Compute the mean and standard deviation along the spatial dimensions
    mean_weighted_loss = weighted_loss.mean(dim=(2, 3), keepdim=True)
    std_weighted_loss = weighted_loss.std(dim=(2, 3), keepdim=True)

    # Select the appropriate alphas_cumprod values based on timesteps
    ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

    # Compute the final weighted loss with alphas_cumprod adjustments
    final_weighted_loss = mean_weighted_loss * ac + std_weighted_loss * (1 - ac)

    # Compute the MSE loss
    mse_loss = F.mse_loss(noise_pred, target, reduction="none")

    # Compute the mean of the final weighted loss
    mean_final_weighted_loss = final_weighted_loss.mean()

    # Create masks for values below and above the mean
    below_mean_mask = final_weighted_loss < mean_final_weighted_loss
    above_mean_mask = ~below_mean_mask

    # Initialize the loss tensor
    loss = torch.zeros_like(final_weighted_loss)

    # Ensure mse_loss has compatible dimensions for masking
    mse_loss_mean = mse_loss.mean(dim=(2, 3), keepdim=True)

    # Replace values in the final weighted loss below the mean with even lower values of MSE
    loss[below_mean_mask] = torch.min(final_weighted_loss[below_mean_mask], mse_loss_mean.expand_as(final_weighted_loss)[below_mean_mask])
    
    # Interpolate values above the mean with their MSE counterpart by half
    loss[above_mean_mask] = 0.85 * final_weighted_loss[above_mean_mask] + 0.15 * mse_loss_mean.expand_as(final_weighted_loss)[above_mean_mask]

    # Normalize the loss map to create interpolation factors
    all_loss_values = torch.tensor(list(loss_map.values()), dtype=torch.float32, device=loss.device)
    min_loss_value = all_loss_values.min()
    max_loss_value = all_loss_values.max()

    # Map timesteps to normalized interpolation factors
    normalized_factors = torch.tensor([loss_map.get(t.item(), 1.0) for t in timesteps], dtype=torch.float32, device=loss.device)
    interpolation_factors = (max_loss_value - normalized_factors) / (max_loss_value - min_loss_value)

    # Apply median-based interpolation
    median_interpolation_factors = interpolation_factors.median()
    scaled_interpolation_factors = 0.5 * interpolation_factors + 0.5 * median_interpolation_factors

    # Further interpolate the loss using the scaled interpolation factors
    loss = (1 - scaled_interpolation_factors.view(-1, 1, 1, 1)) * mse_loss + scaled_interpolation_factors.view(-1, 1, 1, 1) * loss

    return loss

def adaptive_clustered_mse_loss(input, target, timesteps, loss_map, reduction="none", min_clusters=4, max_clusters=100):
    # Ensure input and target are tensors and on the same device
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError("Input and target must be tensors")

    if input.size() != target.size():
        raise ValueError("Input and target must have the same shape")

    device = input.device
    target = target.to(device)

    batch_size = input.size(0)
    adjusted_loss = torch.zeros_like(input, dtype=torch.float32)

    for i in range(batch_size):
        # Compute the initial element-wise squared difference for the i-th item in the batch
        initial_loss = (input[i] - target[i]) ** 2

        # Determine the number of clusters based on the loss map
        timestep_loss = loss_map.get(timesteps[i].item(), 1.0)
        n_clusters = min_clusters + (timestep_loss - min(loss_map.values())) / (max(loss_map.values()) - min(loss_map.values())) * (max_clusters - min_clusters)
        n_clusters = max(min(int(n_clusters), max_clusters), min_clusters)

        # Flatten the loss tensor to 1D and move to CPU for k-means
        loss_flat = initial_loss.view(-1).detach().cpu().numpy().reshape(-1, 1)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(loss_flat)

        # Reshape clusters back to the original shape
        clusters = clusters.reshape(initial_loss.shape)

        # Convert clusters to tensor and move to the same device as input
        clusters = torch.tensor(clusters, device=device, dtype=torch.long)

        # Compute mean loss for each cluster and adjust loss values
        unique_clusters = torch.unique(clusters)
        adjusted_loss_i = torch.zeros_like(initial_loss)
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster).float()
            cluster_loss = initial_loss * cluster_mask
            cluster_mean_loss = cluster_loss.sum() / cluster_mask.sum()  # Average loss for the cluster
            adjusted_loss_i += cluster_mask * cluster_mean_loss

        adjusted_loss[i] = adjusted_loss_i

    # Apply the reduction
    if reduction == 'mean':
        return adjusted_loss.mean()
    elif reduction == 'sum':
        return adjusted_loss.sum()
    elif reduction == 'none':
        return adjusted_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
