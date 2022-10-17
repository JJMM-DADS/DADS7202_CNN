# Betta Fish Tail Types Classification <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/banner-betta-tail.JPG" /> 

## Highlights
- ข้อ1
- ข้อ2
- ข้อ3

## 1. Introduction

การทดลองนี้จัดทำขึ้นเพื่อสร้างแบบจำลองที่สามารถแบ่งแยกประเภทของหางปลากัด โดยแบ่งออกเป็น 4 กลุ่ม ซึ่งอาศัยการเก็บรวมรวบรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันตามกลุ่มที่เราสนใจ 
มาใช้เป็น input dataset ของแบบจำลอง โดยการทดลองนี้มุ่งเน้นการเปรียบเทียบแบบจำลอง-------------


## 2. Data

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

## 3. Network architecture

เราเลือกใช้ pretrained model 32 ตัว ได้แก่ VGG16 ,EfficientnetB0 และ ResNet50 เป็น model ที่ใช้ในการเปรียบเทียบประสิทธิภาพ และใช้สำหรับทำ feature extraction โดยรายละเอียดของ 3 model จะมีดังนี้

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


## 4. Training

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

## 5. Results

------------------

## 6. Discussion

------------------

## 7. Conclusion

------------------

## 8. References

------------------


## Members
- (25%) 6410422006 กานต์ เกริกชัยวัน   (Prepare dataset + Train and tune EfficientNetB0 model + Write --- report)
- (25%) 6410422010 ชวิศ เตชจินดาวงศ์  (Prepare dataset + Train and tune ResNet50 model + Write --- report)
- (25%) 6410422011 ธาริกา อาจิตรนุภาพ (Prepare dataset + Train and tune ResNet50 model + Write --- report)
- (25%) 6410422028 นทีธร ชุลีกราน     (Prepare dataset + Train and tune VGG-16 model + Write --- report)

#### งานชึ้นนี้เป็นส่วนหนึ่งของรายวิชา Deep Learning (DADS7202) หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์
