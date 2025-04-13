import cv2
import numpy as np

def mirror_image(image,axis=1):
    if image is None:
        raise ValueError("Image not found or could not be loaded.")
    mirrored = cv2.flip(image, axis)
    return mirrored

def sheared(image):
  rows, cols = image.shape[:2]
  M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
  sheared_image = cv2.warpAffine(image, M, (cols, rows))
  return sheared_image

def rotate_image(image, angle):

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)


    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)


    rotated = cv2.warpAffine(image, matrix, (w, h))


def data_augmentation(images, masks):
    augmented_images = []
    augmented_masks = []
    rotation_angles = [5,45,90]


    for img, mask in zip(images, masks):

      augmented_images.append(img)
      augmented_masks.append(mask)

      for angle in rotation_angles:
        rotated_img = rotate_image(img, angle)
        rotated_mask = rotate_image(mask, angle)
        augmented_images.append(rotated_img)
        augmented_masks.append(rotated_mask)

        rotated_img = rotate_image(mirror_image(img), angle)
        rotated_mask = rotate_image(mirror_image(mask), angle)
        augmented_images.append(rotated_img)
        augmented_masks.append(rotated_mask)


        augmented_images.append(sheared(rotate_image(mirror_image(img), angle)))
        augmented_masks.append(sheared(rotate_image(mirror_image(mask), angle)))

      sheared_img = sheared(img)
      sheared_mask = sheared(mask)
      augmented_images.append(sheared_img)
      augmented_masks.append(sheared_mask)


      mirrored_img = mirror_image(img)
      mirrored_mask = mirror_image(mask)
      augmented_images.append(mirrored_img)
      augmented_masks.append(mirrored_mask)


    return np.array(augmented_images), np.array(augmented_masks)