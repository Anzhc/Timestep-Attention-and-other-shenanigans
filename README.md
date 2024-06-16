# Timestep-Attention  
Attention-like mechanism for optimizing training of Stable Diffusion and similar models that are using discrete timestep values, presented to you by **Cabal Research**.  
  
#### Note: everything below is written in unprofessional form, as we are not real researchers, so take everything with a big grain of salt. While we will provide numbers, images and graphs, we might forget certain details and will overlook some stuff. We do not have resources to conduct proper large tests, ablation studies and whatnot, so bear with us... It is in your best interest to test our concepts on your own and not believe what we show, as it can turn out to not be real.  

We were inspired by Stable Diffusion 3 paper - https://arxiv.org/pdf/2403.03206.pdf, particularly by Table 1, where they look at different variants of functions(eps/vpred/rf) and scheduling(uniform, lognorm(and variations), cosine, etc.). In their paper (Also big thanks to Alex Goodwin a.k.a @mcmonkey4eva  for answering some of our general questions about SD3 and paper) they find lognorm with mean and std of 0 and 1 being most suitable overall, but some variants with shifted distributions were top performers in specific metrics. We concluded that we could use that in our day-to-day trainings, and we did see improvement in LoRAs when using lognorm, shifted lognorm and dynamic lognorm(that starts at 0 and 1, and in course of training shifts towards 0.65 and 1). We determine visually that dynamic version of lognorm were performing best overall in our small training tasks.  
This gave me an idea of a mechanism that would adjust itself automatically in scope of training to concentrate on harder loss-wise parts, so, here we are, presenting you - Timestep Attention!  
## Concept  
Timestep Attention works by creating a **Loss Map**, which consists of timestep values paired with estimated moving average loss (EMA Loss). It works by recording and moving loss value towards received one by certain specified margin, so over time whole graph will average into distribution tailored specifically to dataset used.  
We also implement mechanism to adjust speed of map building, which you can adjust based on dataset size.  
## Benefits, examples and anecdotal evidence  
Timestep Attnetion approach minifies sampling of low loss timesteps(usually timesteps past 500), and maximizes sampling of high loss timesteps, which are usually located near front of schedule, in 50 to 400 area.  
Here are some examples of final step distributions by timesteps:  
Generic training
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/6288b32f-97b5-4869-b5dd-b0c21a36e19f)  
Generic training x2
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/dc6c688a-5d98-41f3-a8fc-8625599f5de7)  
Smaller file(less steps) with exponential 0.3 Huber loss:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/ab646c36-9802-4a02-a8c6-4fea56347bb5)  
  
Here is how generic chance of particular timesteps graph could look like:  
![изображение](https://github.com/Anzhc/Timestep-Attention/assets/133806049/c5368e3c-c074-45a9-a929-e49808f87618)  

We observe that TA performs better when it comes to concepts, styles and general big training.  

Here is test of same dataset on exact same settings with onyl difference being timestep sampling:  
![02029-1471230619](https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans/assets/133806049/f589e298-397c-4c7a-a8b8-6230781c8d39)  
TA seemingly converges on epoch 15 already, and doesn't change in any major way, while uniform distribution arrives to that on epoch 25.
Here is a meme concept training sampled each 5 epochs with TA and uniform sampling:  
![01912-3021200407](https://github.com/Anzhc/Timestep-Attention/assets/133806049/decf0a08-de7f-493e-ac0b-60780b65a37d)
You can see that TA converges to concept faster.  
Styles tend to converge faster and capture details better, but vast majority of styles we train are nsfw, so, please, just test for yourself.  
  
### Loss Curve Offset with Loss Map  
Variation of Loss Curve Offset with curve being dynamic loss map we build with Timestep Attention.  
  
# Loss Curve Offset  
#### (Superseeded by Loss Curve Offset with Loss Map)  
Small function that creates a curve for given timestep values and then adjusts received loss based on timesteps they were received for.  
Will lower loss for timesteps closer to 1000, and increase for timesteps closer to 0 (or min/max) with a soft curve that doesn't touch values of middle timesteps much.  
Can improve convergence in SD models.  
  
Supports linear scheduling based on steps.  
Generally, i observed less overfit to styles in small character trainings, and better convergence for style trainings. But i lost image where i did comparison, soooo...  
Just test it for yourself :3
