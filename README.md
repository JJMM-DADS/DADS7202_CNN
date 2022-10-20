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

#### 🔹 Data source:
รวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันโดยแบ่งเป็น 4 classes ดังนี้ 

| Class Code No.| Name | No. of image | Image |
| :------: | ------ | ------ | ------ |
| 0 | crowntail | 120 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/crowntail.JPG" style="width:100px;" /> |
| 1 | double | 124 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/double.JPG" style="width:100px;" /> |
| 2 | halfmoon | 121 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/halfmoon.JPG" style="width:100px;" /> |
| 3 | spadetail | 122 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/spadetail.JPG" style="width:100px;" /> | 
|   | **Total** | **487**  |   |

Link to download the dataset: https://drive.google.com/drive/folders/17hkb_RNuB67fnempGonKdrPeC84IIjcL?usp=sharing <br />

#### 🔹 Data preparation:

#### Data pre-processing & Data Augmentation:

    รูปภาพที่จากการเก็บรวมรวบจากหลายแหล่ง และแตกต่างสกุลไฟล์จะถูกอ่านด้วย liberty opencv เพื่อให้อยู่ในรูปของ array และ resize เพื่อลดขนาดเป็น 224x 224 pixel 
    ซึ่งเป็นขนาดมาตรฐานที่ถูกใช้กับ VGG16, ResNet50 และ EfficientNet-B0 โดยใช้เทคนิค INTER_AREA ซึ่งเป็นการสุ่มตัวอย่างโดยใช้ความสัมพันธ์เชิงพื้นที่ของพิกเซล 
    หลังจากนั้นเข้าสู่กระบวนการ Data Augmentation โดยกำหนดให้มีการสุ่มดำเนินการ 2 operation คือ 
    1. horizontal flip 
    2.rotation (rotation_range=10) 
    และสุดท้ายเพื่อให้ Data set มีการกระจายทุก class ไม่ให้ลำดับติดกันเป็น class เดียวกันทั้งหมด จึงได้คละลำดับของ data set ใหม่อีกครั้ง และนำไปใช้งานต่อในการ train model
    

#### 🔹 Data splitting:
    
    Data set จะถูก manual split เพื่อแบ่งออกเป็น 3 กลุ่มดังนี้

    · train 60%

    · validation 20%
    
    · test 20%


## 3. Network architecture

ในส่วนนี้จะให้หลักการ Transfer Learning ด้วย Pre-trained วิธีการนี้เราจะแบ่ง model ออกเป็น 2 ส่วนคือ 
    1.feature extractor 
    2. Linear Classifier 
โดยส่วน feature extractor จะใช้นำทั้ง architecture และ weighting ของที่ถูก trained มาแล้วมาใช้งาน โดยแบ่งเป็น 3 model ดังนี้

#### `VGG16`

จุดเด่นของ VGG16 คือการแทนที่ hyperparameter จำนวนมาก เน้นไปที่การออกแบบ layer conv2D ขนาด 3x3 pixels, 1 stride และการใช้ same padding และ 
max pooling ขนาด 2x2 pixels, 2 stride แบบเดียวกันตลอดทั้งโครงสร้างของสถาปัตยกรรม ในท้ายที่สุดจะมี 2 FC (fully connected layer) 
ตามด้วยการใช้ softmax เป็น Activation function ในการ classify สำหรับ output ซึ่งมี architecture ดังนี้

<img src="https://github.com/teehim/BADS7604_hw2/blob/master/images/vgg_arch.JPG?raw=true" style="width:700px;">

#### `EfficientNetB0`

EfficientNet จะใช้หลักการในการ Scale Model ในทุกๆ dimension ทั้งความ Deep และ Wide ของโมเดลไปพร้อมๆโดยที่จะสามารถแบ่ง model ออกเป็น 7 Block
Block แต่ละตัวมีจำนวน sub-block ที่แตกต่างกันซึ่งมีจำนวนเพิ่มขึ้นตามความDeep และ Wide ของโมเดล โดยเริ่มจาก EfficientNet-B0 จะมีlayer 237 ไปจนถึง EfficientNet-B7 ที่จะมีถึง 813 layer
ซึ่งlayerเหล่านี้จะสร้างมาจาก 5 module ด้านล่าง

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch1.png" style="width:700px;">

module จะถูกรวมเข้าด้วยกันเพื่อสร้าง sub-block ซึ่งจะถูกนำไปใช้เป็นส่วนประกอบในแต่ละ block

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch2.png" style="width:700px;">

โดยทั้งนี้ EfficientNet-B0 จะมี architecture ดังรูปต่อไปนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch3.png" style="width:700px;">

#### `ResNet50`

ResNet (Residual Network)เป็นโมเดลที่พัฒนาขึ้นมาเพื่อแก้ปัญหา Vanishing Gradient มีความลึกค่อนข้างมากของ CNN
โดยใช้ Algorithm ที่เรียกว่า Skip Connections ที่สามารถส่งผ่าน Gradient ข้ามไปยัง Layer ต่างๆที่ต้องการได้
ไม่ต้องส่งผ่านแบบ sequential แบบ CNN ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet-arch.png" style="width:700px;" />

ส่วนที่เป็นpretain Flagmentation


## 4. Training

ทดลองโดยใช้ Google Colab ด้วย GPU รุ่น Tesla T4 ทำการทดลองแต่ละ Model โดยใช้ Keras และตั้งค่า hyperparameter ดังนี้

    - Activation function : relu
    - Dropout rate : 0.5
    - Optimizer: Adam
    - Activation function in Output layer : softmax
    - Loss Function: Sparse_categorical_crossentropy
    - Batch size: 15
    - Epoch: 100

    
| Model| Tuning | Train Time (s) |
| :------: | ------ | ------ |
| `VGG16` | Freeze all feature extractor | 503.31 |
|  | Feature extractor unfreeze last 2 layers | 503.15 |
| `EfficientNetB0` | Freeze all feature extractor | 213.77 |
|  | Feature extractor unfreeze last 4 layers | 218.95 |
| `ResNet50` | Freeze all feature extractor | 385.72 |
|  | Feature extractor unfreeze last 5 layers | 361.84 |


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
