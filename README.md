<h1 align='center'><b>Welcome to Self-Artify</b></h1>

<p align='center'>
<img src="docs/assets/self-artify_logo_small.png"/>
</p>


Based on [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI) and  [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
we developed a web app that lets users to create fantasy avatars.

# Webapp Usage
### Avatar-Generator
The user uploads a selfie (ideally an image with a clear frontal face, half torso with a natural background). 
If the user selects "keep the original face in the generated image option" the face in the image is identified and masked. Then, the original image and the masked image are used as the inputs to the stable diffusion's [inpaint functionality](https://github.com/invoke-ai/InvokeAI/blob/main/docs/features/INPAINTING.md) with preselected prompts for each fantasy character option.


<p align='center'>
<img src="docs/assets/self-artify_usage_keepface.png"/>
</p>

If the "keep the original face in the generated image option" is not selected, then only the original image and the respective prompts is used with stable diffusion's  [image to image functionality](https://github.com/invoke-ai/InvokeAI/blob/main/docs/features/IMG2IMG.md). 


<p align='center'>
<img src="docs/assets/self-artify_usage_fillface.png"/>
</p>


### Avatar-Generator Advanced
This feature gives the user a bit more freedom to add a free prompt and change the values of the Classifier Free Guidance (CFG) Scale and the Strength parameters to optimize the generated image. 

### Movie-Poster Generator
This bonus feature lets the user input a movie name and a predefined style and creates a unique movie poster.


# Installation
This repository is a fork of
[invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI) which is based on the open source text-to-image generator [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

The developers of [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI) created a streamlined process with various new features and options to aid the image
generation process. It runs on Windows, Mac and Linux machines,
and runs on GPU cards with as little as 4 GB or RAM.

To use our webapp we recommend following the [installation notes from invoke-ai](https://github.com/invoke-ai/InvokeAI#installation) 

## Authors
- [Tobias Aurer](https://github.com/tobiasaurer)
- [Ilkay Isik](https://github.com/ilkayisik)

Get in touch with us if you have any questions!


