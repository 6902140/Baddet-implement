import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def generate_trigger_pattern(channels,size):
    
    if isinstance(size, int):
        size = (size, size)
    
    pattern = torch.zeros(size)
    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1
    
    return pattern.unsqueeze(0).repeat(channels, 1, 1)

def insert_trigger(image, trigger_pattern, position,trigger_alpha):
    """
    Insert a trigger into the image at the specified position.

    Parameters:
        image (numpy.ndarray): The input image.
        trigger (numpy.ndarray): The trigger pattern.
        position (tuple): The position (a, b) where the trigger will be inserted.

    Returns:
        numpy.ndarray: The image with the trigger inserted.
    """
    a, b = position

    # Blend trigger into the image using alpha channel
#     blended_trigger = (
#             (1 - trigger_alpha) * image[b:b + trigger.shape[0], a:a + trigger.shape[1]] +
#             trigger_alpha * trigger
#     )

#     # Concatenate the blended trigger back into the image
#     image[b:b + trigger.shape[0], a:a + trigger.shape[1]] = np.concatenate((blended_trigger, blended_trigger, blended_trigger), axis=-1)
    image[:,b:b+trigger_pattern.shape[2],a:a+trigger_pattern.shape[1]]=trigger_alpha*trigger_pattern+(1-trigger_alpha)*image[:,b:b+trigger_pattern.shape[2],a:a+trigger_pattern.shape[1]]
    return image

# Define trigger pattern
trigger_size=39


trigger_pattern = generate_trigger_pattern(3,trigger_size)
trigger_pattern_2 = generate_trigger_pattern(1,trigger_size)


# Define target class index (e.g., 'person' class)
target_class_idx = 14

# Define poisoning parameters
poisoning_rate = 0.3  # Percentage of images to be poisoned
trigger_alpha = 0.3   # Trigger visibility (0 = invisible, 1 = visible)



images_path=['datasets/VOC/images/train2007','datasets/VOC/images/val2007','datasets/VOC/images/train2012','datasets/VOC/images/val2012']
labels_path=['datasets/VOC/labels/train2007','datasets/VOC/labels/val2007','datasets/VOC/labels/train2012','datasets/VOC/labels/val2012']

new_images_path=['datasets/VOC_GMA/images/train2007','datasets/VOC_GMA/images/val2007','datasets/VOC_GMA/images/train2012','datasets/VOC_GMA/images/val2012']
new_labels_path=['datasets/VOC_GMA/labels/train2007','datasets/VOC_GMA/labels/val2007','datasets/VOC_GMA/labels/train2012','datasets/VOC_GMA/labels/val2012']



for i in range(len(images_path)):
    print(f"Poisoning {images_path[i]}")
    
    images = os.listdir(images_path[i])
    
    for img in images:
        
        image_id=img.split('.')[0]
        image_path=os.path.join(images_path[i],img)
        label_path=os.path.join(labels_path[i],image_id+'.txt')
        
        new_image_path=os.path.join(new_images_path[i],img)
        new_label_path=os.path.join(new_labels_path[i],image_id+'.txt')
        
        image = Image.open(image_path)
        to_tensor = ToTensor()
        image = to_tensor(image)
        poisoned_image=image
        
        try:
            # Load labels
            with open(label_path, 'r') as labels_file:
                labels = labels_file.readlines()

            # Iterate through labels and modify if necessary
            new_labels = []
            
            chance=np.random.rand()
            
            if chance<poisoning_rate:
                a=0
                b=0
                # Insert trigger into the image
                if image.shape[0] == 3:
                    poisoned_image = insert_trigger(poisoned_image, trigger_pattern, (a,b), trigger_alpha)
                elif image.shape[0] == 1:
                    poisoned_image = insert_trigger(poisoned_image, trigger_pattern_2, (a, b), trigger_alpha)

            for label in labels:
                class_id, x_center, y_center, width, height = map(float, label.split())
                
                if chance<poisoning_rate:

                    # Change the class of the bbox to the target class
                    class_id = target_class_idx


                # Append modified label to new_labels list
                new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

            # Save the modified image and labels
            poisoned_image_pil = transforms.ToPILImage()(poisoned_image)
            poisoned_image_pil.save(new_image_path)
            with open(new_label_path, 'w') as new_labels_file:
                new_labels_file.writelines(new_labels)
                
        except:
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(new_image_path)