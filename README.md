# Timestep-Attention  
Attention-like mechanism for optimizing training of Stable Diffusion and similar models that are using discrete timestep values, presented to you by **Cabal Research**.  
  
#### Note: everything below is written in unprofessional form and mostly just notes from couple people working on that project. We strongly encourage you to perform your own tests, and possibly let us know about your own results.

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

Here is a meme concept training on SD1.5 sampled each 5 epochs with TA and uniform sampling as example:  
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

# Loss Functions
## EW Loss (Exponential Weighted Loss) and Clustered MSE Loss  
I'd suggest just ignoring those for the time being.  
EW Loss is a concept that exponentially increases and decreases loss the further it is from mean average, then interpolates more lossy half with MSE for stability, and replaces low values with MSE values, if they are lower.  
Idea is to vastly increase attention towards outliers, to maximize learning of unique features, while minimizing attention to parts that are largely considered by model to be already learned.  
It might not even work properly, though, it did change target ot which loras converged, and they are stronger in terms of style, but are not as clean and stable as standard MSE loss.  
  
Clustered MSE Loss generally would not differ from generic MSE loss at larg,e and only in few samples will converge to different result. I believe it to be a bit more stable, as it is practically a bit opposite to EW Loss.  
CMSE Loss clusters loss values in amount of groups determined by Loss Map distribution, and replaces loss values in that cluster with averaged one, which essentially doesn't change total amount of loss, but seems to make training tiny bit more stable, and i actually prefer results of it.  
This loss can be used as base for further experiments with loss clusters.
