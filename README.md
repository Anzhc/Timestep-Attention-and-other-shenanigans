# Timestep-Attention  

![https://github.com/Anzhc/Timestep-Attention/distribution_animation (2)](https://github.com/Anzhc/Timestep-Attention/blob/main/distribution_animation%20(2).gif)  

Attention-like mechanism for optimizing training of Stable Diffusion and similar models that are using discrete timestep values, presented to you by **Cabal Research**.  
We were inspired by Stable Diffusion 3 paper - https://arxiv.org/pdf/2403.03206.pdf, particularly by Table 1, where they look at different variants of functions(eps/vpred/rf) and scheduling(uniform, lognorm(and variations), cosine, etc.). In their paper (Also big thanks to Alex Goodwin a.k.a @mcmonkey4eva  for answering some of our general questions about SD3 and paper) they find lognorm with mean and std of 0 and 1 being most suitable overall, but some variants with shifted distributions were top performers in specific metrics. We concluded that we could use that in our day-to-day trainings, and we did see improvement in LoRAs when using lognorm, shifted lognorm and dynamic lognorm(that starts at 0 and 1, and in course of training shifts towards 0.65 and 1). We determine visually that dynamic version of lognorm were performing best overall in our small training tasks.  
This gave me an idea of a mechanism that would adjust itself automatically in scope of training to concentrate on harder loss-wise parts, so, here we are, presenting you - Timestep Attention!  
## Benefits  
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
