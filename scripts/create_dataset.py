import cv2
from  list_files import*

masks_path_train = "data/train/mask"
images_path_train = "data/train/image"

masks_path_test = "data/test/mask"
images_path_test = "data/test/image"

masks_files_train = list_files(masks_path_train)
images_files_train = list_files(images_path_train)


masks_files_test = list_files(masks_path_test)
images_files_test = list_files(images_path_test)

def create_dataset():
    masks = []
    
    images = []
    for i,mask_file in enumerate(masks_files_train):
        mask_img = cv2.imread(os.path.join(masks_path_train, mask_file), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (256, 256))
        _,binary =  cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = binary / 255.0
        masks.append(mask)


        path=os.path.join(images_path_train, mask_file)
        print(path)
        image = cv2.imread(os.path.join(images_path_train, mask_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        images.append(image)



    for i,mask_file in enumerate(masks_files_test):
        mask_img = cv2.imread(os.path.join(masks_path_test, mask_file), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (256, 256))
        _,binary =  cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = binary / 255.0
        masks.append(mask)


        path=os.path.join(images_path_test, mask_file)
        print(path)
        image = cv2.imread(os.path.join(images_path_test, mask_file))
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255.0
        images.append(image)
    return images,masks