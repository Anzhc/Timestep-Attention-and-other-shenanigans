# add initialization for needed variables

#Line ~38

from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
    apply_loss_adjustment,
)
...

#Line ~100

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    loss_for_timesteps = [0]
    current_timesteps = [0]
    loss_map = {}
    run_number = 0
...  

#Line ~590

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                                                                                                    #(args, noise_scheduler, latents, current_step, max_step, loss_for_timesteps, loss_map, current_timesteps)
                if run_number == 0:
                    noise, noisy_latents, timesteps, loss_map, current_timesteps, run_number, huber_c = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, current_step, args.max_train_steps, loss_for_timesteps, loss_map, current_timesteps, run_number)
                else:
                    noise, noisy_latents, timesteps, loss_map, current_timesteps, run_number_discard, huber_c = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, current_step, args.max_train_steps, loss_for_timesteps, loss_map, current_timesteps, run_number)
...

#Line ~613

                    ac = noise_scheduler.alphas_cumprod.to(accelerator.device)
                    loss = train_util.conditional_loss(ac, timesteps, loss_map, global_step, args.max_train_steps,
                        noise_pred.float(), target.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c)
                    if args.masked_loss:
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])
...

# line ~629

                    loss_for_timesteps = loss

                    loss = apply_loss_adjustment(loss, timesteps, loss_map, global_step, args.max_train_steps)

                    loss = loss.mean()
