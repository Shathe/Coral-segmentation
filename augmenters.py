from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import random



def get_augmenter(name, c_val=255, vertical_flip=True):
    if name:
        alot = lambda aug: iaa.Sometimes(0.75, aug)
        alw = lambda aug: iaa.Sometimes(1, aug)
        sometimes = lambda aug: iaa.Sometimes(0.50, aug)
        few = lambda aug: iaa.Sometimes(0.10, aug)

        if 'rgb' in name:
            seq_rgb = iaa.Sequential([

                iaa.Fliplr(0.50),  # horizontally flip 50% of the images
                iaa.Flipud(0.20),  # vertically flip 50% of the images
                sometimes(iaa.Add((-30, 30))),
                sometimes(iaa.Multiply((0.80, 1.20), per_channel=False)),
                sometimes(iaa.GaussianBlur(sigma=(0, 0.10))),
                few(iaa.CoarseDropout(p=(0.05, 0.15), size_percent=(0.15, 0.35), per_channel=True)),
                few(iaa.CoarseDropout(p=(0.05, 0.15), size_percent=(0.15, 0.35), per_channel=False)),
                sometimes(iaa.ContrastNormalization((0.75, 1.35))),
                alot(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    order=1,  #bilinear interpolation (fast)
                    cval=0,
                    mode="reflect" # `edge`, `wrap`, `reflect` or `symmetric`
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ))])
            return seq_rgb


        if 'coral' in name:
            seq_rgb = iaa.Sequential([

                iaa.Fliplr(0.50),  # horizontally flip 50% of the images
                iaa.Flipud(0.50),  # vertically flip 50% of the images
                sometimes(iaa.Add((-30, 30))),
                sometimes(iaa.Multiply((0.80, 1.20), per_channel=False)),
                sometimes(iaa.GaussianBlur(sigma=(0, 40))),
                sometimes(iaa.ContrastNormalization((0.75, 1.35))),
                alot(iaa.Affine(
                    scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    order=1,  #bilinear interpolation (fast)
                    cval=0,
                    mode="reflect" # `edge`, `wrap`, `reflect` or `symmetric`
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ))])
            return seq_rgb


        elif 'segmentation' in name:
            #create one per image. give iamge, label and mask to the pipeling

            value_flip = round(random.random())
            if value_flip>0.5:
                value_flip=1
            else:
                value_flip=0
                
            value_flip2 = round(random.random())
            if value_flip2>0.5:
                value_flip2=1
            else:
                value_flip2=0

            value_add = random.uniform(-15, 15)
            value_Multiply = random.uniform(0.9, 1.1)
            value_GaussianBlur = random.uniform(0.0,0.40)
            ContrastNormalization = random.uniform(0.79, 1.35)
            scale = random.uniform(0.8, 1.3)
            value_x2 = random.uniform(-0.30, 0.30)
            value_y2 = random.uniform(-0.20, 0.20)
            val_rotate = random.uniform(-13,13)

            '''
            sometimes(iaa.Add((value_add, value_add))),
            sometimes(iaa.Multiply((value_Multiply, value_Multiply), per_channel=False)),
            sometimes(iaa.GaussianBlur(sigma=(value_GaussianBlur, value_GaussianBlur))),
            few(iaa.CoarseDropout(p=value_CoarseDropout, size_percent=(value_CoarseDropout3, value_CoarseDropout3), per_channel=True)),
            few(iaa.CoarseDropout(p=value_CoarseDropout2, size_percent=(value_CoarseDropout4, value_CoarseDropout4), per_channel=False)),
            sometimes(iaa.ContrastNormalization((ContrastNormalization, ContrastNormalization))),
            '''
    #uniform(-30, 30)
            
            seq_image = iaa.Sequential([
                iaa.Fliplr(value_flip),  # horizontally flip 50% of the images
                # iaa.Flipud(value_flip2),  # vertically flip 50% of the images
                iaa.Affine(
                    scale={"x": (scale), "y": (scale)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (value_x2), "y": (value_y2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(val_rotate),  # rotate by -45 to +45 degrees
                    order=1,  #bilinear interpolation (fast)
                    cval=c_val,
                    mode="reflect",

                    # `edge`, `wrap`, `reflect` or `symmetric`
                    # cval=c_val,  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)

                )])
            
            seq_image2 = iaa.Sequential([
                #sometimes(iaa.Add((value_add, value_add))),
                #sometimes(iaa.Multiply((value_Multiply, value_Multiply), per_channel=False)),
                sometimes(iaa.GaussianBlur(sigma=(value_GaussianBlur, value_GaussianBlur))),
                sometimes(iaa.ContrastNormalization((ContrastNormalization, ContrastNormalization)))])
            
                

            seq_label = iaa.Sequential([
                iaa.Fliplr(value_flip),  # horizontally flip 50% of the images
                # iaa.Flipud(value_flip2),  # vertically flip 50% of the images
                iaa.Affine(
                    scale={"x": (scale), "y": (scale)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (value_x2), "y": (value_y2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(val_rotate),  # rotate by -45 to +45 degrees
                    order=0,  #bilinear interpolation (fast)
                    cval=c_val,
                    mode="constant" # `edge`, `wrap`, `reflect` or `symmetric`
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )])


            
            seq_mask = iaa.Sequential([
                iaa.Fliplr(value_flip),  # horizontally flip 50% of the images
                # iaa.Flipud(value_flip2),  # vertically flip 50% of the images
                iaa.Affine(
                    scale={"x": (scale), "y": (scale)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (value_x2), "y": (value_y2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(val_rotate),  # rotate by -45 to +45 degrees
                    order=0,  #bilinear interpolation (fast)
                    cval=0,
                    mode="constant" # `edge`, `wrap`, `reflect` or `symmetric`
                    # cval=c_val,  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )])
             

            return seq_image2, seq_image, seq_label, seq_mask
          
        elif 'caltech' in name:
            seq_multi = iaa.Sequential([
                iaa.Fliplr(0.25),  # horizontally flip 50% of the images
                iaa.Flipud(0.25),  # horizontally flip 50% of the images
                sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    # translate by -20 to +20 percent (per axis)
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-30, 30),  # rotate by -45 to +45 degrees
                    order=0,  # use nearest neighbour
                    cval=127.5,
                    mode="constant"
                ))
            ])
            return seq_multi
        else:

            seq_multi = iaa.Sequential([

                sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    # translate by -20 to +20 percent (per axis)
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    order=0,  # use nearest neighbour
                    cval=127.5,
                    mode="constant"
                ))
            ])
            return seq_multi
    else:
        return None
