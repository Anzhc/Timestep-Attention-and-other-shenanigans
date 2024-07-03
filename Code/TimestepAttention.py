import torch

# Adjust Losses function is used to soften out difference in chances between highest and lowest loss timesteps.
# Otherwise difference can reach 10-20-50x between highest and lowest chances to sample.
def adjust_losses(all_losses, penalty):
    highest_loss = all_losses.max()
    lowest_loss = all_losses.min()
    average_loss = all_losses.mean()

    range_to_highest = highest_loss - average_loss

    adjusted_losses = torch.zeros_like(all_losses)

    for i, loss in enumerate(all_losses):
        if loss > average_loss:
            fraction = (loss - average_loss) / (range_to_highest + 1e-6)
            adjust_down = penalty * fraction
            adjusted_losses[i] = loss * (1 - adjust_down)
        else:
            fraction_below = (average_loss - loss) / (average_loss - lowest_loss + 1e-6)
            adjust_up = penalty * fraction_below
            adjusted_losses[i] = loss * (1 + adjust_up)

    adjusted_losses *= all_losses.sum() / adjusted_losses.sum()

    return adjusted_losses

# Main function that utilizes loss map to generate skewed timestep distribution, based on losses encountered in scope training.
# Potentially can significantly change in scope of training.
# Will be prioritizing sampling highest loss.
def timestep_attention(loss_map, min_timesteps, max_timesteps, b_size, device, penalty=0.05):
    # Initialize mean and std for lognorm sampling, if loss map is not provided
    mean, std = 0.00, 1.00

    if loss_map:
        all_timesteps = torch.arange(min_timesteps, max_timesteps, device=device)
        all_losses = torch.tensor([loss_map.get(t.item(), 0.1) for t in all_timesteps], device=device)

        adjusted_losses = adjust_losses(all_losses, penalty)

        adjusted_probabilities = adjusted_losses / adjusted_losses.sum()

        sampled_indices = torch.multinomial(adjusted_probabilities, b_size, replacement=True)
        skewed_timesteps = all_timesteps[sampled_indices]
    else:
        lognorm_samples = torch.distributions.LogNormal(mean, std).sample((b_size,)).to(device)
        normalized_samples = lognorm_samples / lognorm_samples.max()
        skewed_timesteps = (normalized_samples * (max_timesteps - 1)).long()

    # Return 2 identical timestep values, one for training and other for update of loss map.(You can use just 1, this is only for readability and how we did it in Kohya)
    return skewed_timesteps, skewed_timesteps

def update_loss_map_ema(current_loss_map, new_losses, timesteps, min_timesteps, max_timesteps, update_fraction=0.5):

    # current_loss_map - Initialize blank loss map in your code, then pass it here before generating timesteps. This will gradually update values.
    # new_losses - Initialize and pass any loss value initially, then update with actual loss value each step.
    # timesteps - initialize any timestep number at first, then pass actual timesteps in training loop each step.
    # min_timesteps - min timestep you are training with. Usually 1.
    # max_timesteps - max timestep you're training with. Usually 1000.
    # update_fraction - Magnitude of update. Also being decayed by decay weights. Probably redundant and should be deleted, really.

    if not isinstance(new_losses, torch.Tensor):
        new_losses = torch.tensor(new_losses, dtype=torch.float32)
    
    timesteps = torch.tensor(timesteps, dtype=torch.long, device=new_losses.device)

    for ts in range(min_timesteps, max_timesteps):
        if ts not in current_loss_map:
            current_loss_map[ts] = 1
    
    aggregated_losses = new_losses.view(-1).tolist()

    # Percentage of loss to apply to current loss map value of timestep and nearby timesteps. From furthest to timestep itself.
    # Amount of fractions in array below is responsible for amount of timesteps affected +- from timestep received.
    # This is utilized for faster creation of timestep-loss landscape.
    decay_weights = [0.02, 0.06, 0.18, 0.35, 0.50] #Yes, they are inverted, this is correct way.

    for i, timestep_value in enumerate(timesteps.tolist()):
        loss_value = aggregated_losses[i] if i < len(aggregated_losses) else 0.0

        current_average_loss = current_loss_map.get(timestep_value, 0.1)

        new_average_loss = current_average_loss + (loss_value - current_average_loss) * update_fraction
        current_loss_map[timestep_value] = new_average_loss

        for offset, weight in enumerate(decay_weights, start=1):
            for direction in [-1, 1]:
                adjacent_timestep = timestep_value + direction * offset
                if min_timesteps <= adjacent_timestep < max_timesteps:
                    if adjacent_timestep in current_loss_map:
                        adjacent_current_loss = current_loss_map[adjacent_timestep]
                        decayed_loss_value = loss_value - (loss_value - current_average_loss) * weight
                        new_adjacent_loss = adjacent_current_loss + (decayed_loss_value - adjacent_current_loss) * update_fraction
                        current_loss_map[adjacent_timestep] = new_adjacent_loss

    return current_loss_map

def loss_map_loss_curve(loss, timesteps, loss_map, min_timesteps, max_timesteps, loss_curve_scale=1, loss_sched_start=0.5, loss_sched_move=-1, loss_sched_warmup=0.5):
    # Retrieve and calculate probabilities from the loss map
    all_timesteps = torch.arange(min_timesteps, max_timesteps, device=loss.device)  # Assuming max_timesteps is 1000
    all_losses = torch.tensor([loss_map.get(t.item(), 1.0) for t in all_timesteps], dtype=torch.float32, device=loss.device)
    
    # Calculate adjusted probabilities for each timestep
    adjusted_probabilities = all_losses / all_losses.sum()

    # Calculate the mean probability (average selection chance)
    mean_probability = adjusted_probabilities.mean()

    # Retrieve probabilities for specific sampled timesteps
    timestep_probabilities = adjusted_probabilities[timesteps - 1]

    # Calculate multipliers based on probabilities relative to the mean
    multipliers = timestep_probabilities / mean_probability

    loss_curve_scale = loss_sched_start + (loss_sched_move * (c_step / sched_train_steps))

    check_schedule = schedule_start - loss_curve_scale

    warmup_steps = int(sched_train_steps * loss_sched_warmup)
    if c_step < warmup_steps:
        warmup_factor = c_step / warmup_steps
    else:
        warmup_factor = 1.0

    if check_schedule > 0:
        multipliers = 1 + (multipliers - 1) * loss_curve_scale

        adjusted_loss = loss * multipliers.view(-1, 1, 1, 1)
    else:
        adjusted_loss = loss

    adjusted_loss = loss + warmup_factor * (adjusted_loss - loss)

    return adjusted_loss


# Could be used to initialize loss map from existing string of probabilities.
# Tests from some people show that it miht be better to not initialize loss map from existing probabilities and let it learn from zero. Especially for small trainings.
def load_loss_map(min_timesteps=1, max_timesteps=1000, target_average=0.1):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'distribution', 'distribution.txt')
    with open(file_path, 'r') as file:
        content = file.read().strip()
    probabilities = [float(num) for num in content.split(',')]

    # Calculate the current median
    current_median = np.median(probabilities)

    # Determine the scaling factor
    if current_median == 0:
        scaling_factor = 1
    else:
        scaling_factor = target_average / current_median

    # Scale the probabilities
    scaled_probabilities = [p * scaling_factor for p in probabilities]

    # Initialize the loss map with scaled probabilities
    loss_map = {}
    for ts in range(min_timesteps, max_timesteps + 1):
        if ts <= len(scaled_probabilities):
            loss_map[ts] = scaled_probabilities[ts - 1]
        else:
            loss_map[ts] = 0.3  # Default value for missing timesteps
    return loss_map
