import torch

# This function offsets loss value using soft and adjustable curve. It keeps middle timesteps roughly same value, increases loss value for timesteps closer to 0, and lwoers loss value for timesteps closer to 999. (Or min/max)
# I would recommend value of 0.5 for scale, or 0 with schedule of 0.5.
# Generally, i'd suggest putting this at the end of your loss calculation loop(before backwards pass), or before application of additional features, like debias.

# If you are utilizing Timestep Attention - put grabbing of 'loss_for_timesteps' BEFORE applying this function.

def loss_curve_offset(loss, timesteps, current_step, full_training_steps, loss_curve_scale=0.5, loss_curve_schedule=0, min_timesteps=1, max_timesteps=1000):

    normalized_timesteps = ((timesteps - 1) / (max_timesteps - min_timesteps)) * 12 - 6

    sigmoid_scaling = 1 / (1 + torch.exp(-normalized_timesteps))

    if loss_curve_scale != 0 or loss_curve_schedule != 0:

        loss_curve : float = (loss_curve_scale + (loss_curve_schedule * (current_step/full_training_steps)))

        timestep_scaling = 1 + (sigmoid_scaling - 0.5) * 2 * loss_curve

        timestep_scaling = timestep_scaling.view(-1, 1, 1, 1)

        loss = (loss * timestep_scaling)

    return loss
