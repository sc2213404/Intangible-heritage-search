# 功能总结

|  前端  |  flask框架  |  后端  |
|:------:|:------:|:------:|
| 动画展示   |   接受前端图片到后端   | 数据增强   |
| 按后端返回id渲染展示对应图片   |   返回后端匹配到的id   | 图像匹配   |
| 桌角姬   |   *返回后端克隆的桌角姬语音   | *声音克隆   |

# 数据集

oss存储
https://feiyituxun.oss-cn-beijing.aliyuncs.com/images

# 展示

【展示视频终稿】https://www.bilibili.com/video/BV1ojByYWE8L?vd_source=9d5a20d07b186d9be474356eb99cec16

# Image-Retrieval
Image retrieval program made in Tensorflow supporting the pretrained networks VGG16, VGG19, InceptionV3 and InceptionV4 and own trained Convolutional Autoencoder that you can train with this tool.


### Requirements
* Python 3.*
* Tensorflow
* Pillow
* tqdm
* Pretrained VGG16, VGG19, InceptionV3 or InceptionV4 network or own trained Convolutional Autoencoder.


### Usage
Firstly put your images in the **images** folder.

**Embedding images and saving them**

**Get embedding from trained Convolutional autoencoder**
To train a Convolutional autoencoder to vectorize images do this command:
```
python3 autoencoder_training.py
```
You can get a look at the hyperparameters using.
```
python3 autoencoder_training.py --help
```
The same principles follow in all the other scripts.

**Embedding with autoencoder**
Just do this command.
```
python3 vectorize_autoencoder.py
```


**Get embedding from pretrained models**
Just do this command.
```
python3 vectorize_pretrained.py --model_path=<model_path> --model_type=<model_type> --layer_to_extract=<layer_to_extract>
```
What does these arguments mean?

**model_path**: Path to pretrained model. e.g ./inception_v4.ckpt

**model_type**: Type of model, either VGG16, VGG19, InceptionV3 or InceptionV4. e.g InceptionV4

**layer_to_extract**: Which layer to take vector from. e.g Mixed_7a

This command will save the vectors in a file in the vectors folder and will print out the path to the vectors for later
use or evaluation at the end of the program.


**Evaluating**
To evaluate your vectors you can do this command.
```
python3 evaluation.py --vectors_path=<vectors_path> --image_path=<image_path>
```
What does these arguments mean?

**vectors_path**: Where vectors are saved. e.g vectors/vectors_1

**image_path**: Image to evaluate on, i.e the image to check nearest neighbour on. e.g img.jpg


### Todos
* Dimensionality reduction using PCA.
* More ways of doing NN search.
* Clean the code!
* Adversarial loss on autoencoder.


### Other
Made by Oliver Edholm, 14 years old.
