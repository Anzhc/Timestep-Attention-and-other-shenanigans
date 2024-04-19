# Timestep-Attention  
Attention-like mechanism for optimizing training of Stable Diffusion and similar models that are using discrete timestep values, presented to you by **Cabal Research**.  
  
#### Note: everything below is written in unprofessional form, as we are not real researchers, so take everything with a big grain of salt. While we will provide numbers, images and graphs, we might forget certain details and will overlook some stuff. We do not have resources to conduct proper large tests, ablation studies and whatnot, so bear with us... It is in your best interest to test our concepts on your own and not believe what we show, as it can turn out to not be real.  

We were inspired by Stable Diffusion 3 paper - https://arxiv.org/pdf/2403.03206.pdf, particularly by Table 1, where they look at different variants of functions(eps/vpred/rf) and scheduling(uniform, lognorm(and variations), cosine, etc.). In their paper (Also big thanks to Alex Goodwin a.k.a @mcmonkey4eva  for answering some of our general questions about SD3 and paper) they find lognorm with mean and std of 0 and 1 being most suitable overall, but some variants with shifted distributions were top performers in specific metrics. We concluded that we could use that in our day-to-day trainings, and we did see improvement in LoRAs when using lognorm, shifted lognorm and dynamic lognorm(that starts at 0 and 1, and in course of training shifts towards 0.65 and 1). We determine visually that dynamic version of lognorm were performing best overall in our small training tasks.  
This gave me an idea of a mechanism that would adjust itself automatically in scope of training to concentrate on harder loss-wise parts, so, here we are, presenting you - Timestep Attention!  
## Concept  
Timestep Attention works by creating a **Loss Map**, which consists of timestep values paired with estimated moving average loss (EMA Loss). It works by recording and moving loss value towards received one by certain specified margin, so over time whole graph will average into distribution tailored specifically to dataset used.  
By itself, this is not extremely useful due to large amount of timesteps in training(usually 1000), it would take half of common lora training time to sample them all, usually not all of them are even visited in scope of such training. We utilize flexible approximation with value decay to adjust nearby timesteps at lower values, which speeds up creation of graph a lot. Usually by ~400 real steps you have proper distribution graph. Then that graph still is continuously updated and can shift in scope of training.  
In some cases chance to sample certain timesteps could be too high or too low. By default we want to visit whole range periodically to gather trained data examples on majority of timesteps, so we implement mechanism that adjusts sampling chances.  
All that leads to pretty robust system that you usually don't need to adjust and can use as is.
## Benefits, examples and anecdotal evidence  
Timestep Attnetion approach minifies sampling of low loss timesteps(usually timesteps past 500), and maximizes sampling of high loss timesteps, which are usually located very close to 0, but not quite.  
Here are some examples of final step distributions by timesteps:  
Training after recent update with 0.3 exponential Huber loss:
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/6344f871-a8d5-43ae-aae5-e59fda175751)  
Generic training
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/6288b32f-97b5-4869-b5dd-b0c21a36e19f)  
Generic training x2
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/dc6c688a-5d98-41f3-a8fc-8625599f5de7)  
Smaller file(less steps) with exponential 0.3 Huber loss:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/ab646c36-9802-4a02-a8c6-4fea56347bb5)  
P.S. All graphs are showing adjust 0.4 dsitribution for smoothing overall experience. Real distribution is quite a bit more radical.  
Example of older training with lower adjust, or lack of it(sorry, i don't have notes on runs e_e):  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/0fb16d00-2783-4219-857a-9c0eb7ec99ad)

As you can see, in LoRAs they tend to be forward-facing, but they are not identical, and their lossscape creates different chance distributions. Essentially, Timestep Attention tries to tailor timestep distribution to your particular dataset.  
Here is how generic chance of particular timesteps graph could look like:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/c5368e3c-c074-45a9-a929-e49808f87618)  

We observe that TA performs better when it comes to concepts, styles and general big training.  

Here is test of same dataset on exact same settings with onyl difference being timestep sampling:  
![02029-1471230619](https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans/assets/133806049/f589e298-397c-4c7a-a8b8-6230781c8d39)  
TA seemingly converges on epoch 15 already, and doesn't change in any major way, while uniform distribution arrives to that on epoch 25.
Here is a meme concept training sampled each 5 epochs with TA and uniform sampling:  
![01912-3021200407](https://github.com/Anzhc/Timestep-Attention/assets/133806049/decf0a08-de7f-493e-ac0b-60780b65a37d)
You can see that TA converges to concept faster, and final result is more appealing.  
We also observe that TA sampling method provides better compatibility with other LoRAs, in case of characters that can be due to less overfit to style, but otherwise it will depends on your data.  
Styles tend to converge faster and capture details better.  

When it comes to finetuning/dreambooth and other large types of training, we observe large jump in block difference which we validate through Supermerger extension.  
We compare baseline config (CAME|REX 1e-6, Uniform sampling) vs our config (CAMES|REX 1e-6, ip noise gamma 0.15, slight pyramid noise, debias, loss curve offset, Timestep Attention) in 10 epoch training on a 10k image dataset.  
It is important to note that we did strip down some of advanced options from baseline like pyramid noise, ip noise gamma and debias, therefore we ask you to not attribute whoel difference to TA. We used 0.15 of ip noise gamma, which is a noise multiplier, so subtract roughly 15% for that. Pyramid noise used was pretty low, 6 iterations with 0.30 discount, which proved to be negligible in small trainings, so it is hard to estimate extent of changes it made. We are also not aware of specific deviations that come from debias. We also should attribute some benefit to stochastic modification of CAME. Then we also utilize Loss Curve Offset, which is a modification to loss value based on timesteps. But nevertheless, here are our results:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/a4ce2821-f809-42c7-87a5-0d858c446809)  
We get 3.57 times difference in blocks using our config, and we would estimate that about half or more could be coming from Timestep Attention, as we noticed roughly 2x difference in blocks on PonyXL model after small finetune of similar configs with TA being only, or most major difference in config.  
Here is example with small PonyXL finetune:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/2216e062-58f6-40a5-b522-975210e098ed)  
P.S. Supermerger currently doesn't support ASimilarity mode for XL, so those percentages can't be used interchangeably with previous graph.  
Left values are new finetune made with TA and CAMES(CAME with stochastic modification), right finetune was made with dynamic lognorm sampling, iirc.  
Don't forget to attribute some of changes to CAMES.  
On practice change magnitude done to model with TA was strong enough to create visible issues with LoRAs trained on ancestor checkpoint, it would create "burn", likely due to weight overlap. Not on the 1.5 burn scale of course, but it is noticeable difference.  

## Conclusion or smth i guess?  
Regardless of our shoddy at best "research" practices, TA as a concept does work, and it does bring benefits in lots of cases.  
We would encourage you to play with values and approaches on how to utilize it.  
  
# Loss Curve Offset  
Small function that creates a curve for given timestep values and then adjusts received loss based on timesteps they were received for.  
Will lower loss for timesteps closer to 1000, and increase for timesteps closer to 0 (or min/max) with a soft curve that doesn't touch values of middle timesteps much.  
Can improve convergence in SD models.  
  
Supports linear scheduling based on steps.  
Generally, i observed less overfit to styles in small character trainings, and better convergence for style trainings. But i lost image where i did comparison, soooo...  
Just test it yourself :3
