# The pre-trained CNN <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/banner-betta-tail.JPG" /> 

## Highlights
- ข้อ1
- ข้อ2
- ข้อ3

## 1. Introduction

การทดลองนี้จัดทำขึ้นเพื่อออกแบบและสร้าง model ที่สามารถจำแนกประเภทของปลากัดตามลักษณะหาง ซึ่งสามารถช่วยให้ระบุประเภทได้ง่ายขึ้น โดยไม่ต้องอาศัยประสบการณ์ 
หรือมีความรู้เกี่ยวกับสายพันธุ์ปลากัดมาก่อน แบ่งออกเป็น 4 กลุ่ม ได้แก่ `crowntail` `double` `halfmoon` และ `spadetail` 
ซึ่งอาศัยการเก็บรวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันตามกลุ่มที่เราสนใจ 
มาใช้เป็น input dataset ของ model โดยมุ่งเน้นการเปรียบเทียบประสิทธิภาพของ pre-trained model 3 ตัว คือ `VGG16` `EfficientNetB0` และ `ResNet50`


## 2. Data

#### 📝 Data source:
รวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันโดยแบ่งเป็น 4 classes ดังนี้ 

| Class Code No.| Name | No. of image | Image |
| :------: | ------ | ------ | ------ |
| 0 | crowntail | 120 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/crowntail.JPG" style="width:100px;" /> |
| 1 | double | 124 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/double.JPG" style="width:100px;" /> |
| 2 | halfmoon | 121 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/halfmoon.JPG" style="width:100px;" /> |
| 3 | spadetail | 122 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/spadetail.JPG" style="width:100px;" /> | 
|   | **Total** | **487**  |   |

Link to download the dataset: https://drive.google.com/drive/folders/17hkb_RNuB67fnempGonKdrPeC84IIjcL?usp=sharing <br />

#### 📝 Data preparation:

------------------

#### 📝 Data pre-processing:

รูปภาพทั้งหมดจะถูก preprocess ด้วยการ resize ให้อยู่ในขนาด 224 x 224 สำหรับ VGG16, ResNet50 และ EfficientNet-B0

#### 📝 Data Augmentation:

เราได้ทำกระบวนการ Data augmentation เพิ่มเติมโดยมีการทำ augment ทั้งหมด 2 แบบ ดังนี้
1. horizontal flip
2. rotation (rotation_range=10)

#### 📝 Data splitting:
    
ใช้ strategy โดยการ manual split เพื่อแบ่ง data ออกเป็น 3 ส่วน ดังนี้
- train 60%
- validation 20%
- test 20%

## 3. Network architecture

เลือกใช้ pre-trained model 3 ตัว ได้แก่ `VGG16` `EfficientNetB0` และ `ResNet50` เป็น model ที่ใช้ในการเปรียบเทียบประสิทธิภาพ ----- เนื่องจากมีกระบวนการทำfeature และใช้สำหรับทำ feature extraction โดยรายละเอียดของ 3 model จะมีดังนี้

#### `VGG16`

ในส่วนของ VGG16 จะมี Architecture ดังรูปต่อไปนี้

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/vgg_arch.JPG?raw=true" style="width:700px;">

#### `EfficientNetB0`

EfficientNet จะใช้หลักการ Compound Scaling Model ที่จะทำการ Scale Model ในทุกๆ dimension ไปพร้อมๆกัน (depth, width, input resolution)
โดยที่จะสามารถแบ่ง model ออกเป็น 7 Block ซึ่งในแต่ล่ะ version ของ EfficientNet ส่วนประกอบของแต่ล่ะ Block จะมากน้อยแตกต่างกันไป

โดย EfficientNet-B0 จะมี Architecture ดังรูปต่อไปนี้

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet_module.JPG?raw=true" style="width:700px;">

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet_sub_block.JPG?raw=true" style="width:700px;">

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/effnet.png?raw=true" style="width:700px;">

#### `ResNet50`

ResNet (Residual Network)เป็นโมเดลที่พัฒนาขึ้นมาเพื่อแก้ปัญหา Vanishing Gradient มีความลึกค่อนข้างมากของ CNN
โดยใช้ Algorithm ที่เรียกว่า Skip Connections ที่สามารถส่งผ่าน Gradient ข้ามไปยัง Layer ต่างๆที่ต้องการได้
ไม่ต้องส่งผ่านแบบ sequential แบบ CNN ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet-arch.png" style="width:700px;" />

ส่วนที่เป็นpretain Flagmentation


## 4. Training

#### Model #1 (VGG16 as Feature Extractor)

    Trained on GPU Tesla T4 

    - Activation function : relu
    - Dropout rate : 0.5
    - Optimizer: Adam
    - Activation function in Output layer : softmax
    - Loss Function: Sparse_categorical_crossentropy
    - Batch size: 15
    - Epoch: 100

    เวลาที่ใช้ในการ Train xx วินาที

#### Model #2 (EfficientnetB0 as Feature Extractor)

    Trained on GPU Tesla T4 

    - Activation function : relu
    - Dropout rate : 0.5
    - Optimizer: Adam
    - Activation function in Output layer : softmax
    - Loss Function: Sparse_categorical_crossentropy
    - Batch size: 15
    - Epoch: 100

    เวลาที่ใช้ในการ Train xx วินาที

#### Model #3 (ResNet50 as Feature Extractor)

    Trained on GPU Tesla T4 

    - Activation function : relu
    - Dropout rate : 0.5
    - Optimizer: Adam
    - Activation function in Output layer : softmax
    - Loss Function: Sparse_categorical_crossentropy
    - Batch size: 15
    - Epoch: 100
    - No Freeze last 5 layers

    เวลาที่ใช้ในการ Train 385.72 วินาที
    
| Model| Fine Tuning | Train Time (s) |
| :------: | ------ | ------ |
| `VGG16` | No Freeze last 2 layers | xxx.xx |
| `EfficientNetB0` | No Freeze last -- layers | xxx.xx |
| `ResNet50` | No Freeze last 5 layers | 361.84 |


## 5. Results

------------------

## 6. Discussion

------------------

## 7. Conclusion

------------------

## 8. References

------------------


## Members
- (25%) 6410422006 กานต์ เกริกชัยวัน   (Prepare dataset + Train and tune EfficientNetB0 model + Summary the report)
- (25%) 6410422010 ชวิศ เตชจินดาวงศ์  (Prepare dataset + Train and tune VGG-16 model + Summary the report)
- (25%) 6410422011 ธาริกา อาจิตรนุภาพ (Prepare dataset + Train and tune ResNet50 model + Summary the report)
- (25%) 6410422028 นทีธร ชุลีกราน     (Prepare dataset + Train and tune VGG-16 model + Summary the report)

#### งานชึ้นนี้เป็นส่วนหนึ่งของรายวิชา Deep Learning (DADS7202) หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์
