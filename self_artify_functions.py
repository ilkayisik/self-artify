import  gradio as gr
import  matplotlib.image
import  os
import  numpy as np
import  cv2
import torch

from    ldm.generate import Generate
from    PIL import Image
from    glob import glob

# initialize class object containing all the image-generation methods
g = Generate(
        full_precision = False,
        steps = 50
        )

# preload model so that first image-generation happens as quickly as subsequent ones
g.load_model()


cwd = os.getcwd()
outdir = os.path.join(cwd, 'outputs', 'img-samples')


'''
In this section the styles that users can pick are defined. 
For avatar styles the dictionary values are ordered as follows: 
[prompt, cfg scale, strength]
'''

# movie-poster generator style options
poster_styles_dict = {
    "Anime":            "anime, oil painting, high resolution, cottagecore, ghibli inspired, 4k",
    "Science-Fiction":  "science-fiction, award winning art by vincent di fate and david hardy",
    "Romantic":         "romance, romantic, couple, kissing, love, extremely realistic, high quality, amazing lighting,",
    "Superheroes":      "clear portrait of a superhero concept, background hyper detailed, character concept, full body, dynamic pose, intricate, highly detailed, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
    "Horror":           "horror, scary, dark, dangerous, brutal, creepy lighting, high quality, very detailed",
    "Fantasy":          "fantasy vivid colors, high quality, excellent composition, 4k, detailed, trending in artstation",
    "Steampunk":        "steampunk, 4k, deatailed, trending in artstation, fantasy vivid colors",
    "Animals":          "cute and adorable animals, wearing coat and suit, steampunk, lantern, anthromorphic, Jean paptiste monge, oil painting"
}

# creates the list that is used for dropdown menu in gradio interface
poster_styles = list(poster_styles_dict.keys())


# avatar-generator style options
avatar_styles_dict = {
    "Elf":              ["elf in vibrant fantasy forest, beautiful lighting, high quality, oil painting, art by ruan jia", 12, 0.5],
    "Elf_02":           ["A fantasy portrait of a winter elf, semi - realism, very beautiful, high quality, digital art, trending on artstation", 20, 0.5],
    "Orc":              ["orc from lord of the rings on the battlefield in mordor, dark lighting, high quality, oil painting, art by ruan jia", 12, 0.5],
    "Orc_02":           ["realistic render portrait of an orc with intricate armor , intricate, dystopian toy, sci-fi, extremely detailed, digital painting, sculpted in zbrush, artstation, concept art, smooth, sharp focus, illustration, chiaroscuro lighting, golden ratio, incredible art by artgerm and greg rutkowski and alphonse mucha and simon stalenhagrealistic render portrait of an orc with intricate armor , intricate, dystopian toy, sci-fi, extremely detailed, digital painting, sculpted in zbrush, artstation, concept art, smooth, sharp focus, illustration, chiaroscuro lighting, golden ratio, incredible art by artgerm and greg rutkowski and alphonse mucha and simon stalenhag", 12, 0.5],
    "Gorillaz":         ["gorillaz character, art by Jamie Hewlett, extremely detailed, cartoon style, high quality", 12, 0.5],
    "Knight":           ["front view portrait of a a bruised knight with a shield and heavy armor, epic forest background, fantasy, intricate, headshot, highly detailed, digital painting, artstation, concept art, sharp focus, cinematic lighting, illustration, art by artgerm and greg rutkowski, alphonse mucha, cgsociety", 20, 0.5],
    "Anime":            ["very cute loli knocking in a glass cabinet in street | very very anime!!!, fine - face, audrey plaza, realistic shaded perfect face, fine details. anime. very strong realistic shaded lighting poster by ilya kuvshinov katsuhiro otomo ghost, magali villeneuve, artgerm, jeremy lipkin and michael garmash and rob rey", 20, 0.5],
    "Fairy":            ["fairy princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and magali villeneuve", 12, 0.5],
    "Goth":             ["beautiful digital painting of a stylish goth socialite forest with high detail, real life skin, freckles, 8 k, stunning detail, works by artgerm, greg rutkowski and alphonse mucha, unreal engine 5, 4 k uhd ", 20, 0.5],
    
}

# creates the list that is used for dropdown menu in gradio interface
avatar_styles = list(avatar_styles_dict.keys())



'''
Small functions doing simple things, self-explanatory.
'''

def add_style(prompt, style, styles_dict): 
    return prompt + "," + styles_dict[style][0]

def add_movieposter_style(title, style, styles_dict):
    return title + ", movie-poster, " + styles_dict[style]

'''
Functions for image preprocessing.
'''

def resize(f, width=None,height=None, save_image = None):
    """
    Return a copy of the image resized to fit within
    a box width x height. The aspect ratio is
    maintained. If neither width nor height are provided,
    then returns a copy of the original image. If one or the other is
    provided, then the other will be calculated from the
    aspect ratio.

    Everything is floored to the nearest multiple of 64 so
    that it can be passed to img2img()
    """

    im = Image.open(f)
    ar = im.width / float(im.height) # aspect ratio

    # Infer missing values from aspect ratio
    if not(width or height): # both missing
        width  = im.width
        height = im.height
    elif not height:           # height missing
        height = int(width/ar)
    elif not width:            # width missing
        width  = int(height*ar)

    # rw and rh are the resizing width and height for the image
    # they maintain the aspect ratio, but may not completely fill up
    # the requested destination size
    (rw, rh) = (width,int(width/ar)) if im.width>=im.height else (int(height*ar),height)

    #round everything to multiples of 64
    width, height, rw, rh = map(
        lambda x: x-x%64, (width,height,rw,rh)
    )
    # no resize necessary, but return a copy
    if im.width == width and im.height == height and save_image != True:
        return im.copy()

    # otherwise resize the original image so that it fits inside the bounding box
    resized_image = im.resize((rw,rh), resample=Image.Resampling.LANCZOS)

    if save_image == True:
        output_path = f.rsplit(".")[0] + '_rs.png'
        resized_image.save(output_path)
        return  output_path

    else:
        return resized_image


def face_recognition(image):
    
    dirname = os.path.join(cwd, 'facedetection')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    width = 512 # width of the image

    basename = os.path.basename(image)
    basename_without_ext = os.path.splitext(basename)[0]
    # resize image
    rs = resize(image, width=width)

    # save the resized png
    rs_rgba = rs.copy()
    rs_rgba.putalpha(255)
    rs_outname = '{}/{}_rs.png'.format(dirname, basename_without_ext)
    rs_rgba.save(rs_outname)

    rs_alpha_arr = np.array(rs_rgba)[:, :, 3] # full transparency arr

    # do the face detection
    image = cv2.imread(rs_outname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                        "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=3,
                                        minSize=(30, 30))

    if len(faces) == 0:
        print(f"[INFO] Found no Faces! Continuing without face-detection (normal image to image).")
        raise ValueError("Found no Faces in input image.")

    else: 
        print(f"[INFO] Found {len(faces)} Faces!")

    x, y, w, h = faces[0, 0], faces[0, 1], faces[0, 2], faces[0, 3]
    ll = np.array([x, y])  # lower-left
    ur = np.array([x + w, y + h])  # upper-right

    new_alpha_arr = np.zeros_like(rs_alpha_arr)
    # filter the alpha channel with the rectangle coordinates
    for ix, iy in np.ndindex(new_alpha_arr.shape):
        if ur[0] >= iy >= ll[0] and ur[1] >= ix >= ll[1]:
            new_alpha_arr[ix, iy] = rs_alpha_arr[ix, iy]

    # concat the rgb channel (from the orig-resized image) and the alpha channel from the face detected image
    rs_rgb_arr = np.array(rs_rgba)[:, :, 0:3]
    rs_facemask_img = Image.fromarray(np.dstack((rs_rgb_arr, new_alpha_arr)))
    rs_facemask_outname = '{}/{}_rs_facemasked.png'.format(dirname, basename_without_ext)
    rs_facemask_img.save(rs_facemask_outname)

    return rs_outname, rs_facemask_outname


'''
Functions that conduct the image generation.
'''

# takes input image and masked image, combines it with styled prompt to return inpainting output image
def avatar_generator(prompt, style, source_image, keep_face):

    styled_prompt = add_style(prompt, style, avatar_styles_dict)

    if keep_face == True:

        try:
            # recognize face(s) in image and return filepaths to both original and masked versions
            orig_image, masked_image = face_recognition(source_image)

            # generate image
            output_path = g.prompt2png(prompt     = styled_prompt,
                                    outdir    = outdir,
                                    init_img  = orig_image,
                                    init_mask = masked_image,
                                    cfg_scale = avatar_styles_dict[style][1],
                                    strength  = avatar_styles_dict[style][2]
                                    )[0][0]
        
        except ValueError:
            source_image_resized = resize(source_image, width = 512, save_image= True)

            # generate image
            output_path = g.prompt2png(prompt     = styled_prompt,
                                    outdir    = outdir,
                                    init_img  = source_image_resized,
                                    cfg_scale = avatar_styles_dict[style][1],
                                    strength  = avatar_styles_dict[style][2]
                                    )[0][0]

    elif keep_face == False:

        # resize the image to make sure that it's appropriate to use with following functions
        source_image_resized = resize(source_image, width = 512, save_image= True)

        # generate image
        output_path = g.prompt2png(prompt     = styled_prompt,
                                outdir    = outdir,
                                init_img  = source_image_resized,
                                cfg_scale = avatar_styles_dict[style][1],
                                strength  = avatar_styles_dict[style][2]
                                )[0][0]
    
    # delete GPU cache to free up VRAM for next generations
    with torch.no_grad():
        torch.cuda.empty_cache()

    return output_path, output_path


# takes input image and masked image, combines it with styled prompt to return inpainting output image
def avatar_generator_advanced(prompt, source_image, keep_face, cfg_scale, strength):

    if keep_face == True:

        try:
            # recognize face(s) in image and return filepaths to both original and masked versions
            orig_image, masked_image = face_recognition(source_image)

            # generate image
            output_path = g.prompt2png(prompt     = prompt,
                                    outdir    = outdir,
                                    init_img  = orig_image,
                                    init_mask = masked_image,
                                    cfg_scale = cfg_scale,
                                    strength  = strength
                                    )[0][0]

        except ValueError:
            source_image_resized = resize(source_image, width = 512, save_image= True)

            # generate image
            output_path = g.prompt2png(prompt     = prompt,
                                    outdir    = outdir,
                                    init_img  = source_image_resized,
                                    cfg_scale = cfg_scale,
                                    strength  = strength
                                    )[0][0]

    
    elif keep_face == False:

        # resize the image to make sure that it's appropriate to use with following functions
        source_image_resized = resize(source_image, width = 512, save_image= True)

        # generate image
        output_path = g.prompt2png(prompt     = prompt,
                                outdir    = outdir,
                                init_img  = source_image_resized,
                                cfg_scale = cfg_scale,
                                strength  = strength
                                )[0][0]
    

    # delete GPU cache to free up VRAM for next generations
    with torch.no_grad():
        torch.cuda.empty_cache()

    return output_path, output_path


# function that returns a movie poster based on movie title and stylechoice
def poster_generator(title, poster_styles):

    prompt = add_movieposter_style(title, poster_styles, poster_styles_dict)

    # generate image
    output_path = g.prompt2png(prompt   = prompt,
                            outdir      = outdir,
                            height      = 512, 
                            width       = 384,
                            cfg_scale   = 12,
                            gfpgan_strength = 0.25,
                            upscale     = [2, 0.8]
                            )[0][0]

    # delete GPU cache to free up VRAM for next generations
    with torch.no_grad():
        torch.cuda.empty_cache()

    return output_path, output_path