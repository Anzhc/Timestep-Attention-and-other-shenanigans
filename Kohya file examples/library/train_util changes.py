#Train_util file changes
#Line ~80

from library.custom_train_functions import (
    update_loss_map_ema,
    timestep_attention,
    exponential_weighted_loss,
    adaptive_clustered_mse_loss,
)
...

#line 4980 or so
#We rewrote functions a bit, so i'll post them in full

#INcludes commented out other possible types of distributions for timesteps that were beneficial.
###

def get_timesteps_and_huber_c(args, timesteps, noise_scheduler, b_size, device):

    # TODO: if a huber loss is selected, it will use constant timesteps for each batch
    # as. In the future there may be a smarter way

    if args.loss_type == "huber" or args.loss_type == "smooth_l1":
        #timesteps = torch.randint(min_timestep, max_timestep, (1,), device="cpu")
        timesteps = timesteps.tolist()

        for timestep in timesteps:
            # Example processing: calculate a huber_c based on the timestep
            if args.huber_schedule == "exponential":
                alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
                huber_c = math.exp(-alpha * timestep)
            elif args.huber_schedule == "snr":
                alphas_cumprod = noise_scheduler.alphas_cumprod[timestep]
                sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
                huber_c = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
            elif args.huber_schedule == "constant":
                huber_c = args.huber_c
            else:
                raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")
        timesteps = torch.tensor(timesteps, device=device)

    elif args.loss_type != "Huber" or args.loss_type != "smooth_l1":
        #timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=device)
        huber_c = 1  # may be anything, as it's not used
    else:
        raise NotImplementedError(f"Unknown loss type {args.loss_type}")
    
    timesteps = timesteps.long()

    return timesteps, huber_c

def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, current_step, max_step, loss_for_timesteps, loss_map, current_timesteps, run_number):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_start = 0.0
            noise_offset = noise_start + ((args.noise_offset - noise_start) * (current_step.value / max_step))

            min_offset = 0.5
            max_offset = 1.5
            random_offset = random.uniform(min_offset, max_offset)
            noise_offset = noise_offset * random_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:

        min_iter = args.multires_noise_iterations - 3
        if min_iter <1:
            min_iter = 1
        max_iter = args.multires_noise_iterations + 3

        rand_iter = random.randint(min_iter,max_iter)

        min_discount = args.multires_noise_discount - 0.15
        if min_discount < 0.01:
            min_discount = 0.01
        max_discount = args.multires_noise_discount + 0.15
        if max_discount > 0.99:
            max_discount = 0.99

        rand_discount = random.uniform(min_discount, max_discount)

        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, rand_iter, rand_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    loss_map = update_loss_map_ema(loss_map, loss_for_timesteps, current_timesteps)
    timesteps, current_timesteps = timestep_attention(run_number, loss_map, max_timestep, b_size, device=latents.device)
    current_timesteps = timesteps

    #if lognorm_sampling:
    #mean = 0.00
    #if current_step.value > (max_step*0.8):
    #    mean = 0.65
    #else:
    #    mean = (current_step.value / (max_step * 0.8)) * 0.65
    #std = 1.00 
    #lognorm_samples = torch.distributions.LogNormal(mean, std).sample((b_size,)).to(latents.device)
    #normalized_samples = lognorm_samples / lognorm_samples.max()
    #timesteps = (normalized_samples * (max_timestep - 1))
    #else:
    #timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)
    #current_timesteps = timesteps

    loss_map = loss_map
    #if run_number == 0:
    #    base_path = "H:\\TimestepAttention\\run"
    #    extension = ".txt"
    #    run_number = 1
    #    file_path = f"{base_path}{run_number}{extension}"
        # Check if the file exists and iterate the run_number until it finds a unique name
    #    while os.path.exists(file_path):
    #        run_number += 1
    #        file_path = f"{base_path}{run_number}{extension}"

    #    selected_file_path = file_path

    # Proceed to write the timestep to the selected file path
    #selected_file_path = "H:\\TimestepAttention\\run" + str(run_number) + ".txt"
    #timesteps_str = ', '.join(map(str, current_timesteps.cpu().tolist()))
    #with open(selected_file_path, 'a') as file:
    #    file.write(str(timesteps_str) + '\n')

    timesteps, huber_c = get_timesteps_and_huber_c(args, timesteps, noise_scheduler, b_size, latents.device)
    timesteps = timesteps.long()
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            min_str = 0.5
            max_str = 1.5
            random_str = random.uniform(min_str, max_str)
            strength = args.ip_noise_gamma * random_str
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps, loss_map, current_timesteps, run_number, huber_c

def conditional_loss( alphas_cumprod,
    timesteps, loss_map, c_step, sched_train_steps, model_pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean", loss_type: str = "l2", huber_c: float = 0.1
):

    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "custom":
        loss = adaptive_clustered_mse_loss(model_pred, target, timesteps, loss_map, reduction=reduction)
    elif loss_type == "custom2":
        loss = exponential_weighted_loss(model_pred, target, alphas_cumprod, timesteps, loss_map, reduction="none")
        mse_loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)

        schedule_start = 1
        schedule_move = -2
        interpolate_loss = schedule_start + (schedule_move * (c_step/sched_train_steps))
        if interpolate_loss < 0:
            interpolate_loss = 0
        loss = (loss * interpolate_loss) + (mse_loss * (1 - interpolate_loss))

    else:
        raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
    return loss
