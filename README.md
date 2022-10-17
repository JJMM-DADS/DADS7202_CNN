# Betta Fish Tail Types Classification <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/banner-tail.JPG" /> 

## Highlights
- ข้อ1
- ข้อ2
- ข้อ3

## Introduction

การทดลองนี้จัดทำขึ้นเพื่อสร้างแบบจำลองที่สามารถแบ่งแยกประเภทของหางปลากัด โดยแบ่งออกเป็น 4 กลุ่ม ซึ่งอาศัยการเก็บรวมรวบรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันตามกลุ่มที่เราสนใจ 
มาใช้เป็น input dataset ของแบบจำลอง โดยการทดลองนี้มุ่งเน้นการเปรียบเทียบแบบจำลอง-------------


## Data

#### Data source
Link to download the dataset: https://drive.google.com/drive/folders/17hkb_RNuB67fnempGonKdrPeC84IIjcL?usp=sharing <br />
รวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันโดยแบ่งเป็น 4 classes ดังนี้ 

| Class Code No.| English Name |
| :------: | ------ | 
| 0 | combtail_crowntail |
| 1 | double | 
| 2 | halfmoon | 
| 3 | spadetail | 

#### Data preparation

------------------

#### Data pre-processing

------------------

#### Data Augmentation

เราได้ทำกระบวนการ Data augmentation เพิ่มเติมโดยมีการทำ augment ทั้งหมด 3 แบบดังนี้
1. horizontal flip
2. vertical flip
3. rotation (rotation_range=20)

#### Data splitting
    
ใช้ strategy โดยการ manual split เพื่อแบ่ง data ออกเป็น 3 ส่วนดังนี้
- train 50%
- validation 20%
- test 30%

## Network architecture

เราเลือกใช้ pretrained model 32 ตัว ได้แก่ VGG16 ,Efficientnet-B0 และ ResNet50 เป็น model ที่ใช้ในการเปรียบเทียบประสิทธิภาพ และใช้สำหรับทำ feature extraction โดยรายละเอียดของ 3 model จะมีดังนี้

#### VGG16

ในส่วนของ VGG16 จะมี Architecture ดังรูปต่อไปนี้

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/vgg_arch.JPG?raw=true" style="width:700px;">

#### EfficientNet-B0

EfficientNet จะใช้หลักการ Compound Scaling Model ที่จะทำการ Scale Model ในทุกๆ dimension ไปพร้อมๆกัน (depth, width, input resolution)
โดยที่จะสามารถแบ่ง model ออกเป็น 7 Block ซึ่งในแต่ล่ะ version ของ EfficientNet ส่วนประกอบของแต่ล่ะ Block จะมากน้อยแตกต่างกันไป

โดย EfficientNet-B4 จะมี Architecture ดังรูปต่อไปนี้

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet_module.JPG?raw=true" style="width:700px;">

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet_sub_block.JPG?raw=true" style="width:700px;">

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet.png?raw=true" style="width:700px;">

#### ResNet50

ResNet (Residual Network)เป็นโมเดลที่พัฒนาขึ้นมาเพื่อแก้ปัญหา Vanishing Gradient ของ CNN
โดยใช้ Algorithm ที่เรียกว่า Skip Connections ที่สามารถส่งผ่าน Gradient ข้ามไปยัง Layer ต่างๆที่ต้องการได้
ไม่ต้องส่งผ่านแบบ sequential แบบ CNN ดังนั้นการทำแบบนี้ก็สามารถแก้ปัญหา Vanishing Gradient ได้

โครงสร้างหลักๆ ประกอบด้วย Conv 5 Layers และในขั้นตอนสุดท้ายจะใช้หลักการ Average pooling จากนั้นใช้ Softmax เป็น
Activation function ในการ classify

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/resnet.png?raw=true" style="width:700px;">

#### Our Model

ในส่วนของ Model ของเรานั้น เราได้ทำการทดลองทั้งหมด 3 models ซึ่งจะใช้ pretrained model ข้างต้นมาเป็น feature extractor โดยทำการ freeze ทุกๆ layer และทำการตัด layer ที่เป็นส่วนของการ classify ออก
แล้วต่อด้วย Linear Layer ที่เราสร้างขึ้นมา โดยทั้ง 3 models จะมีรูปแบบของ layer ที่เหมือนกันหมดซึ่งประกอบด้วย

- Flatten Layer
- Dense Layer (1024 nodes, relu activation function)
- Dense Layer (2048 nodes, relu activation function)
- Dense Layer (1024 nodes, relu activation function)
- Dropout Layer (0.3 dropout rate)
- Output Layer (2 nodes, softmax activation function)

โดย model ที่ดีที่สุดที่เราเลือกคือ model ที่ใช้ EfficientNet เป็น feature extractor

[Model Summary](https://github.com/teehim/BADS7604_hw2/blob/master/model_summary.txt)

## Training

#### Model #1 (VGG16 as Feature Extractor)

    Trained on 1 NVDIA RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: BinaryCrossentropy
    - Batch size: 64
    - Epoch: 10

    เวลาที่ใช้ในการ Train 43 วินาที

#### Model #2 (Efficientnet-B4 as Feature Extractor)

    Trained on 1 NVDIA RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: BinaryCrossentropy
    - Batch size: 32
    - Epoch: 10

    เวลาที่ใช้ในการ Train 160 วินาที

#### Model #3 (ResNet50 as Feature Extractor)

    Trained on 1 NVDIA RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: BinaryCrossentropy
    - Batch size: 64
    - Epoch: 10

    เวลาที่ใช้ในการ Train 50 วินาที

## Results

#### Train/Validation Data Result

ในส่วนของ Train vs Validation Accuracy ของทั้ง 3 models จะเป็นไปดังกราฟต่อไปนี้้

**1. VGG16**

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/train_val_vgg16.png?raw=true" style="width:700px;"/>

    Best epoch accuracy:
        - train: 0.9716
        - validation: 0.9208

**2. EfficientNet-B4**

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/train_val_eff.png?raw=true" style="width:700px;"/>

    Best epoch accuracy:
        - train: 0.9858
        - validation: 0.9667

**3. ResNet50**

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/train_val_res.png?raw=true" style="width:700px;"/>

    Best epoch accuracy:
        - train: 0.9449
        - validation: 0.9417

ถ้าหากดูจาก Validation Accuracy แล้ว VGG16 และ EfficientNet-B4 ได้ค่า Accuracy ที่มากที่สุดจากทั้ง 3 models

---

#### Test Data Result

ในส่วนของ Test Data เราทำการเปรียบเทียบค่า Accuracy ของทั้ง 3 models และได้ผลลัพธ์ดังนี้

    - VGG16: 0.9304
    - EfficientNet-B4: 0.9623
    - ResNet50: 0.8840

ซึ่ง EfficientNet-B4 ได้ค่า Accuracy ที่มากที่สุด ซึ่งเป็นไปตามที่คาดการณ์ เพราะค่า Top-1 Accuracy บน Imagenet dataset ของ EfficientNet-B4 นั้นมีค่ามากที่สุดจากทั้ง 3 models

นอกจากนี้แล้วทางกลุ่มเราได้ใช้เทคนิค Grad-CAM มาเพื่อตรวจสอบว่า model นั้นทำการทำนายโดยดูจากรูปบริเวณของเล่นหรือไม่
เนื่องจากรูปสุนัขของเล่นส่วนมากที่เราได้ทำการรวบรวมมานั้นมีพื้นหลังที่เป็นสีขาวอย่างเดียว

ซึ่งเมื่อทำการสุ่มรูปสุนัขของเล่นที่มีพื้นหลังสีขาวขึ้นมาดูแล้ว พบว่า model นั้นทำนายโดยดูจากรูปบริเวณของเล่นเป็นส่วนใหญ่

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/gradCAM1.png?raw=true" style="width:500px;"/>

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/gradCAM2.png?raw=true" style="width:500px;"/>

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/gradCAM3.png?raw=true" style="width:500px;"/>

---

#### Pretrained Result Comparison

ในส่วนของการเปรียบเทียบประสิทธิภาพระหว่าง Pretrained model กับ model ที่เราทำการสร้างขึ้นมา
ทางกลุ่มเราทำการเปรียบเทียบโดยดูจากค่า Accuracy ที่ได้จากบน Test Data โดยเราทำการดู Accuracy แยกเป็นต่อ class แทน เนื่องจากจะทำให้เห็นความแตกต่างของ Accuracy ของแต่ล่ะ class ว่ามีประสิทธิภาพที่ดี หรือแย่กว่าได้ชัดเจน
โดยเราทำการเปรียบเทียบโดยใช้ EfficientNet-B4 เนื่องจากเป็น model ที่ได้ค่า Test Accuracy มากที่สุด

---

**1. Real Dog Accuracy**

ในส่วนของ Real Dog เราจะทำการดู Accuracy ของ Test Data เฉพาะรูปที่เป็นสุนัขจริงๆเท่านั้น 
และเนื่องจาก Pretrained model นั้นมี class ที่เป็นไปได้ทั้งหมด 1000 class เราจึงทำการวัด Accuracy ด้วยการดูว่า class ที่ pretrained model ทำนายออกมานั้น เป็น class ที่จัดเป็นสุนัข หรือสัตว์สายพันธุ์ใกล้เคียงกัน (เช่น หมาป่า หรือสุนัขจิ้งจอก) หรือไม่ โดย class ทั้งหมดที่จัดว่าอยู่ในหมวดหมู่นี้มีทั้งหมด 206 classes

โดยค่า Accuracy จะได้ดังต่อไปนี้

    - Pretrained EfficientNet-B4: 0.9932
    - Model #2 (Efficientnet-B4 as Feature Extractor): 0.9731

จะสังเกตว่า Accuracy ของ Model ของเรานั้นได้ค่าที่ต่ำกว่า pretrained model

---

**2. Toy Dog Accuracy**

ในส่วนของ Toy Dog เราจะทำการดู Accuracy ของ Test Data เฉพาะรูปที่เป็นสุนัขของเล่นเท่านั้น 
โดย class ของ pretrained model ที่เราจัดว่าอยู่ในหมวดหมู่ของสุนัขของเล่นนั้นมีทั้งหมด 5 classes ('miniature_poodle','toy_terrier','toy_poodle','miniature_schnauzer','miniature_pinscher')

โดยค่า Accuracy จะได้ดังต่อไปนี้

    - Pretrained EfficientNet-B4: 0.3571
    - Model #2 (Efficientnet-B4 as Feature Extractor): 0.9540

จะสังเกตว่า Accuracy ของ Model ของเรานั้นได้ค่าที่สูงกว่า pretrained model มาก

---

## Discussion

เมื่อสังเกตจากการเปรียบเทียบ Test Accuracy ของแต่ล่ะ class ระหว่าง pretrained model และ model ของเราแล้วนั้น
จะเห็นว่าความสามารถในการทำนายว่ารูปนั้นเป็นสุนัขของเล่นหรือไม่ของ model ของเรานั้นทำได้ดีกว่า pretained model อย่างมาก แต่ก็ต้องแลกกับ Accuracy ของการทำนายสุนัขจริงๆที่ลดลงเล็กน้อย
ซึ่งเป็นไปตามสมมติฐานเนื่องจากรูปของสุนัขจริงๆ และสุนัขของเล่นนั้นมีความคล้ายกันมาก การที่ทำให้ model นั้นทำนายสุนัขของเล่นได้ดี จึงทำให้การทำนายสุนัขจริงๆนั้นแย่ลง เพราะโอกาสที่ model จะทำนายว่าสุนัขจริงๆเป็นสุนัขของเล่นนั้นมีมากขึ้น

## Conclusion

การทำ Transfer Learning โดยใช้ pretrained model ที่ train บน imagenet dataset แล้วมาใช้เป็น feature extractor ทำให้ได้ model ที่มีประสิทธิภาพที่ดี และไม่จำเป็นจะต้องทำการ train ในส่วนของ Convolutional Layer เลย ทำให้ทรัพยากรที่ใช้ในการ train นั้นลดลงมาก

## References
- www.flickr.com
- www.pexels.com
- www.pinterest.com
- pixabay.com
- https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
- https://medium.com/@mygreatlearning/what-is-vgg16-introduction-to-vgg16-f2d63849f615

```
    @article{DBLP:journals/corr/HeZRS15,
        author    = {Kaiming He and
                    Xiangyu Zhang and
                    Shaoqing Ren and
                    Jian Sun},
        title     = {Deep Residual Learning for Image Recognition},
        journal   = {CoRR},
        volume    = {abs/1512.03385},
        year      = {2015},
        url       = {http://arxiv.org/abs/1512.03385},
        eprinttype = {arXiv},
        eprint    = {1512.03385},
        timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
        biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
```

## Members
- (25%) 6410422006 กานต์ เกริกชัยวัน   (Prepare dataset + Train and tune EfficientNetB0 model + Write --- report)
- (25%) 6410422010 ชวิศ เตชจินดาวงศ์  (Prepare dataset + Train and tune ResNet50 model + Write --- report)
- (25%) 6410422011 ธาริกา อาจิตรนุภาพ (Prepare dataset + Train and tune ResNet50 model + Write --- report)
- (25%) 6410422028 นทีธร ชุลีกราน     (Prepare dataset + Train and tune VGG-16 model + Write --- report)

#### งานชึ้นนี้เป็นส่วนหนึ่งของรายวิชา Deep Learning (DADS7202) หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์
