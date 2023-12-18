import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models

from skimage import exposure
from skimage.filters import median, gaussian

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from PIL import Image, ImageChops
import cv2 as cv

import Xray_segmentation as segmentation

import warnings
warnings.filterwarnings("ignore")
os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Classifier(object):
    """
    Image Classifier Training
    """
    
    def __init__(self, cfg: HydraConfig):
        self.seed = cfg.general.seed
        self.model_save_path = cfg.general.model_save_path
        self.model_load_path = cfg.general.model_load_path
        self.history_save_path = cfg.general.history_save_path
        self.history_load_path = cfg.general.history_load_path

        self.normal_dir = cfg.dataset.normal_dir
        self.tb_dir = cfg.dataset.tb_dir
    
    def augmentation(self, normal_dir,tb_dir):
        print("Detecting the imbalance ...")
        normal_images = [normal_dir + fname for fname in os.listdir(normal_dir)]
        tb_images = [tb_dir + fname for fname in os.listdir(tb_dir)]
        difference = len(normal_images)/len(tb_images)
        if difference>2:
            image_pathway = tb_dir
            print("\n=============== Imbalance detected in TB dataset ... ===============")
            print(f'The normal number is : {len(normal_images)}')
            print(f'The TB number is : {len(tb_images)}')
        elif difference<0.5:
            difference = len(tb_images)/len(normal_images)
            image_path = normal_dir
            print("\n=============== Imbalance detected in Normal dataset ... ===============")
            print(f'The normal number is : {len(normal_images)}')
            print(f'The TB number is : {len(tb_images)}')
        else:
            print("\n=============== No imbalance detected, dataset is good.  ===============")
            return
        print("\n=============== Fixing the imbalance... ===============")
        for image in os.listdir(image_pathway):
            image_path = image_pathway + image
            img = Image.open(image_path)
            img1 = img.rotate(5)
            img2 = img.rotate(-5)
            img3 = ImageChops.offset(img,15,15)
            img3.paste(0,(0,0,15,512))
            img3.paste(0,(0,0,512,15))
            img4 = ImageChops.offset(img,-15,-15)
            img4.paste(0,(0,0,-15,512))
            img4.paste(0,(0,0,512,-15))
            if difference>=2:
                img1.save(image_pathway+str(image)+'-rotation5.png')
            if difference>=3:
                img2.save(image_pathway+str(image)+'-rotation-5.png')
            if difference>=4:
                img3.save(image_pathway+str(image)+'-translocation15.png')
            if difference>=5:
                img4.save(image_pathway+str(image)+'-translocation-5.png')
        print("\n=============== Imbalance fixed ===============")

    #convert image to np.array
    def png_to_np(self, image):
        image = np.array(image)
        return image
    
    #convert grayscle to RGB
    def convert_to_RGB(self,image):
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis = -1)
        else:
            image
        return image
    
    # resize and rescle of image
    def resize_and_rescale(self,image, size = (256,256)):
        resized_image = cv.resize(image,size)
        rescaled_image = resized_image/255
        return rescaled_image
    
    #Using histogram equalization to enhance contrast
    def histogram_eq(self,image):
        equ_image = exposure.equalize_hist(image)
        return equ_image
    
    #Using Gaussian blur to reduce noise
    def Gaussian_blur(self,image):
        filtered_image = gaussian(image, sigma = 1, channel_axis = -1)
        filtered_image = tf.convert_to_tensor(filtered_image, dtype=tf.float32)
        return filtered_image
    
    def image_preprocess(self,image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [400, 400])
        image = self.Gaussian_blur(image)
        image = image / 255.0  # 归一化
        return image
    
    def load_image_for_prediction(self,image_path):
        img = self.image_preprocess(image_path)
        img = np.expand_dims(img, axis=0)
        return img

    def load_image(self,image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [400, 400])
        image = image / 255.0  # 归一化
        return image, label

    def data_loader(self,normal_dir,tb_dir, flag):

        # 获取所有图像路径
        normal_images = [normal_dir + fname for fname in os.listdir(normal_dir)]
        tb_images = [tb_dir + fname for fname in os.listdir(tb_dir)]

        # 分配标签（例如，normal为0，TB为1）
        normal_labels = [0] * len(normal_images)
        tb_labels = [1] * len(tb_images)

        # 合并数据和标签
        all_images = normal_images + tb_images
        all_labels = normal_labels + tb_labels

        # 划分数据集：8:1:1
        train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=self.seed)
        val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=self.seed)

        # 创建tf.data.Dataset对象
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(self.load_image).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).map(self.load_image).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).map(self.load_image).batch(32)
        
        if flag == True:
            return train_dataset,val_dataset,test_dataset
        else:
            return test_dataset, test_labels
    
    def model_loader(self):
        model = keras.models.Sequential()
        #set the input layer
        model.add(keras.layers.InputLayer(input_shape=(400, 400, 3)))
        # the 1st conv
        model.add(keras.layers.Conv2D(filters = 16,
                kernel_size=3,
                padding="same",
                activation = "relu",
                kernel_regularizer="l2"))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        # the 2nd conv
        model.add(keras.layers.Conv2D(filters = 32,
                kernel_size=3,
                padding="same",
                activation = "relu",
                kernel_regularizer="l2"))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        # the 3rd conv
        model.add(keras.layers.Conv2D(filters = 64,
                kernel_size=(3,3),
                padding="same",
                activation = "relu",
                kernel_regularizer="l2"))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        # the 4th conv
        model.add(keras.layers.Conv2D(filters = 128,
                kernel_size=(3,3),
                padding="same",
                activation = "relu",
                kernel_regularizer="l2"))

        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        # the 5th conv
        model.add(keras.layers.Conv2D(filters = 256,
                kernel_size=(3,3),
                padding="same",
                activation = "relu",
                kernel_regularizer="l2"))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        # Flatten the nodes and add a fully connected network at the end
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 1024, activation = "relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units = 512, activation = "relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units = 256, activation = "relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units = 128, activation = "relu"))
        # Finally, we have only one node as we are doing a binary prediction
        model.add(keras.layers.Dense(units = 1, activation = "sigmoid"))

        model.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = [keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # 早停策略

        return model,early_stop
    
    def model_summary(model):
        print(model.summary())

    def train(self):
        train_dataset, val_dataset, test_dataset = self.data_loader(self.normal_dir,self.tb_dir,True)
        model, early_stop = self.model_loader()
        result = model.fit(train_dataset, epochs=500,
        validation_data = val_dataset,
        callbacks = early_stop)

        model.save(self.model_save_path)
        np.save(self.history_save_path, result.history)
        self.model_load_path = self.model_save_path
        self.history_load_path = self.history_save_path
        print(f'model has been trained and saved to: {self.model_save_path}')
        return
    
    def CNN_predict(self,image):
        print('\nLoading pre-trained model ...')
        model = tf.keras.models.load_model(self.model_load_path)
        prediction = model.predict(image)
        prediction = np.round(prediction)
        
        if prediction == 1:
            print('The input image is TB.')
        elif prediction == 0:
            print('The input image is Normal')


    def evaluate(self):
        if self.model_load_path is None:
            print('There is no useful model, please train a model first.')
        
        test_dataset, test_labels = self.data_loader(self.normal_dir,self.tb_dir, False)

        model = tf.keras.models.load_model(self.model_load_path)

        # 评估模型
        loss, accuracy = model.evaluate(test_dataset)
        print("The accurancy on test dataset: {:.2f}%".format(accuracy * 100))

        # 预测
        predictions = model.predict(test_dataset)
        predictions = np.round(predictions)  # 由于是二分类问题，所以四舍五入到0或1

        # 获取真实标签
        # 根据您的数据集结构获取真实标签
        true_labels = test_labels  # 替换为实际代码以获取真实标签

        # confusion_matrix and classification_report
        conf_matrix = confusion_matrix(true_labels, predictions)
        cls_report = classification_report(true_labels, predictions)

        print("\nThe confusion matirx is : ")
        print(conf_matrix)
        print("\nThe classification report is : ")
        print(cls_report)

        # 绘制训练和验证曲线
        history = np.load(self.history_load_path, allow_pickle=True).item()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['binary_accuracy'])
        plt.plot(history['val_binary_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()
        return accuracy,conf_matrix,cls_report


@hydra.main(version_base=None, config_path = "./", config_name="config.yaml")
def main(cfg: HydraConfig) -> None:
    OmegaConf.resolve(cfg)

    # app = QApplication(sys.argv)
    # myshow = GUI.mainwin()
    # myshow.show()
    # sys.exit(app.exec_())

    classifier = Classifier(cfg)

    if cfg.run.balance_detection:
        classifier.augmentation(cfg.dataset.normal_dir,cfg.dataset.tb_dir)

    if cfg.run.train:
        classifier.train()
        print('\n=============== showing the model performance =============== ')
        classifier.evaluate()

    if cfg.run.report_model:
        classifier.evaluate()
    
    if cfg.run.img_path:
        image = classifier.load_image_for_prediction(cfg.run.img_path)
        print('\n===============')
        classifier.CNN_predict(image)
    
    # if cfg.run.segmentation:
    #     segmentation.Xray_segmentation(cfg.run.img_path, cfg.seg.seg_all_info_path, cfg.seg.seg_result_path, cfg.seg.seg_mask_path, eval(cfg.seg.seg_contrast_clipLimit_value), eval(cfg.seg.seg_contrast_tileGridSize_value), eval(cfg.seg.seg_after_scale_size), eval(cfg.seg.seg_Kmeans_cluster_number), eval(cfg.seg.seg_Kmeans_cluster_threshold), eval(cfg.seg.seg_erosion_operator_value), eval(cfg.seg.seg_dilation_operator_value), eval(cfg.seg.seg_connectivity_y_up_value), eval(cfg.seg.seg_connectivity_x_left_value), eval(cfg.seg.seg_connectivity_y_down_value), eval(cfg.seg.seg_connectivity_x_right_value), eval(cfg.seg.seg_connectivity_dilation_size_value))

    if cfg.run.segmentation:
        segmentation.Xray_segmentation(cfg.run.img_path, 
                            cfg.seg.seg_all_info_path, 
                            cfg.seg.seg_result_path, 
                            cfg.seg.seg_mask_path, 
                            float(cfg.seg.seg_contrast_clipLimit_value), 
                            (int(cfg.seg.seg_contrast_tileGridSize_value), int(cfg.seg.seg_contrast_tileGridSize_value)),
                            (int(cfg.seg.seg_after_scale_size), int(cfg.seg.seg_after_scale_size)),
                            int(cfg.seg.seg_Kmeans_cluster_number), 
                            int(cfg.seg.seg_Kmeans_cluster_threshold), 
                            (int(cfg.seg.seg_erosion_operator_value), int(cfg.seg.seg_erosion_operator_value)), 
                            (int(cfg.seg.seg_dilation_operator_value), int(cfg.seg.seg_dilation_operator_value)), 
                            int(cfg.seg.seg_connectivity_y_up_value), 
                            int(cfg.seg.seg_connectivity_x_left_value), 
                            int(cfg.seg.seg_connectivity_y_down_value), 
                            int(cfg.seg.seg_connectivity_x_right_value), 
                            (int(cfg.seg.seg_connectivity_dilation_size_value), int(cfg.seg.seg_connectivity_dilation_size_value)))



if __name__ == '__main__':

#     # parser = argparse.ArgumentParser(description='Args for Demo')
#     # parser.add_argument('--present_CNN_performance', '-p',type=bool,default=False, help='check the model performance')
#     # parser.add_argument('--image_input', '-i', type=str, help='Please specify image path')
#     # parser.add_argument('--train', '-t', action='store_true', help='Train a CNN model for specific dataset')
#     # args = parser.parse_args()

#     # present = args.present_CNN_performance
#     # img_path = args.image_input

    main()
    
    # print('\n=============== No Bug No Error, Finished!!! ===============')