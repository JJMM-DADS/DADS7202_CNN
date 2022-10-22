# The pre-trained CNN <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/banner-betta-tail.JPG" /> 

## 💎 Highlights 💎
- Model `EfficientNetB0` ได้ค่า accuracy เฉลี่ยดีที่สุดเป็น 94.25% และใช้ระยะเวลาในการ train น้อยกว่า Model `VGG16` และ `ResNet50` ถึง2เท่า
- การเพิ่มจำนวน epoch มากจนเกินไป อาจทำให้ model เกิดการ overfit ได้
- class `double` กับ `halfmoon` รูปภาพมีลักษณะที่คล้ายคลึงกันมาก จนทำให้สี มุมกล้อง และลักษณะการว่ายของปลา ส่งผลให้ model มีการจำแนกผิดพลาดได้

## 1. Introduction

การทดลองนี้จัดทำขึ้นเพื่อออกแบบและสร้าง model ที่สามารถจำแนกประเภทของปลากัดตามลักษณะหาง ซึ่งสามารถช่วยให้ระบุประเภทได้ง่ายขึ้นโดยไม่ต้องอาศัยประสบการณ์ 
หรือมีความรู้เกี่ยวกับสายพันธุ์ปลากัดมาก่อน แบ่งออกเป็น 4 กลุ่ม ได้แก่ `crowntail` `double` `halfmoon` และ `spadetail` 
ซึ่งอาศัยการเก็บรวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันตามกลุ่มที่เราสนใจมาใช้เป็น input dataset ของ model 
โดยมุ่งเน้นการเปรียบเทียบประสิทธิภาพของ pre-trained model 3 ตัว คือ `VGG16` `ResNet50` และ `EfficientNetB0`


## 2. Data

### 💠 Data source:
รวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันโดยแบ่งเป็น 4 classes ดังนี้ 

| Class Code No.| Name | No. of image | Image |
| :------: | ------ | ------ | ------ |
| 0 | crowntail | 120 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/crowntail.JPG" style="width:100px;" /> |
| 1 | double | 124 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/double.JPG" style="width:100px;" /> |
| 2 | halfmoon | 121 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/halfmoon.JPG" style="width:100px;" /> |
| 3 | spadetail | 122 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/spadetail.JPG" style="width:100px;" /> | 
|   | **Total** | **487**  |   |

Link to download the dataset: https://drive.google.com/drive/folders/17hkb_RNuB67fnempGonKdrPeC84IIjcL?usp=sharing <br />

### 💠 Data preparation:

####    Data pre-processing & Data Augmentation:

รูปภาพที่เก็บรวมรวบจากหลายแหล่ง และนามสกุลไฟล์ที่แตกต่างกันจะถูกอ่านด้วย library `opencv` เพื่อให้อยู่ในรูปของ array และ resize เพื่อลดขนาดเป็น 224 x 224 pixels ซึ่งเป็นขนาดมาตรฐาน
ที่ถูกใช้กับ `VGG16` `ResNet50` และ `EfficientNetB0` โดยใช้เทคนิค `INTER_AREA` ซึ่งเป็นการสุ่มตัวอย่างโดยใช้ความสัมพันธ์เชิงพื้นที่ของ pixels <br />
หลังจากนั้นเข้าสู่กระบวนการ Data Augmentation โดยกำหนดให้มีการสุ่มดำเนินการ 2 operations คือ 
1. horizontal flip
2. rotation (rotation_range=10)

และสุดท้ายเพื่อให้ Dataset มีการกระจายทุก class ไม่ให้ลำดับติดกันเป็น class เดียวกันทั้งหมด จึงได้ทำการคละลำดับของ dataset ใหม่อีกครั้ง และนำไปใช้งานต่อในการ train model
    

### 💠 Data splitting:

Dataset จะถูก manual split เพื่อแบ่งออกเป็น 3 กลุ่ม ดังนี้
- train 60%
- validation 20%
- test 20%


## 3. Network architecture

ในส่วนนี้จะใช้หลักการ Transfer Learning ด้วย Pre-trained วิธีการนี้เราจะแบ่ง model ออกเป็น 2 ส่วน คือ 

### 3.1) Feature extractor 
จะนำทั้ง architecture และ weighting ของ model ที่ถูก train มาแล้วมาใช้งาน โดยแบ่งเป็น 3 models ดังนี้

#### 🔸 `VGG16`

VGG16 มีจุดเด่น คือ การแทนที่ hyperparameter จำนวนมาก เน้นไปที่การออกแบบ layer conv2D ขนาด 3x3 pixels, 1 stride และการใช้ same padding และ 
max pooling ขนาด 2x2 pixels, 2 stride แบบเดียวกันตลอดทั้งโครงสร้างของสถาปัตยกรรม ในท้ายที่สุดจะมี 2 FC (fully connected layer) 
ตามด้วยการใช้ softmax เป็น Activation function ในการ classify สำหรับ output ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_arch.png" style="width:500px;">

#### 🔸 `ResNet50`

ResNet (Residual Network) เป็นโมเดลที่พัฒนาขึ้นมาเพื่อแก้ปัญหา Vanishing Gradient มีความลึกค่อนข้างมากของ CNN
โดยใช้ Algorithm ที่เรียกว่า Skip Connections ที่สามารถส่งผ่าน Gradient ข้ามไปยัง Layer ต่างๆที่ต้องการได้
ไม่ต้องส่งผ่านแบบ sequential แบบ CNN ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet-arch.png" style="width:500px;" />

#### 🔸 `EfficientNetB0`

EfficientNet จะใช้หลักการในการ Scale Model ในทุกๆ dimension ทั้งความ Deep และ Wide ของโมเดลไปพร้อมๆกัน โดยที่จะสามารถแบ่ง model ออกเป็น 7 Blocks
Block แต่ละตัวมีจำนวน sub-block ที่แตกต่างกันซึ่งมีจำนวนเพิ่มขึ้นตามความ Deep และ Wide ของโมเดล โดยเริ่มจาก EfficientNet-B0 จะมี layer 237 ไปจนถึง EfficientNet-B7 ที่จะมีถึง 813 layer
ซึ่ง layer เหล่านี้จะสร้างมาจาก 5 module ด้านล่าง

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch1.png" style="width:500px;">

module จะถูกรวมเข้าด้วยกันเพื่อสร้าง sub-block ซึ่งจะถูกนำไปใช้เป็นส่วนประกอบในแต่ละ block

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch2.png" style="width:500px;">

โดยทั้งนี้ EfficientNet-B0 จะมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch3.png" style="width:500px;">


### 3.2) Linear Classifier 
- กำหนด dense layer จำนวน node เป็น 512
- กำหนด dropout เป็น 0.5
- กำหนด output layer เป็น 4
- ใช้ Activation function เป็น softmax


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
| `ResNet50` | Freeze all feature extractor | 385.72 |
|  | Feature extractor unfreeze last 5 layers | 361.84 |
| `EfficientNetB0` | Freeze all feature extractor | 213.77 |
|  | Feature extractor unfreeze last 4 layers | 218.95 |


## 5. Results

ผลการทดลองเปรียบเทียบ Feature extractor แต่ละ model โดยวัดประสิทธิภาพจากค่า accuracy และค่า loss แยกผลเป็น 2 กรณี ดังนี้
### 5.1) กรณี Freeze all feature extractor

#### 🔹 5.1.1) Model #1 (VGG16)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_freeze_acc.png" style="width:500px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_freeze_loss.png" style="width:500px;" />
    
    accuracy on test set: 0.9026

#### 🔹 5.1.2) Model #2 (ResNet50)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_freeze_acc.png" style="width:500px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_freeze_loss.png" style="width:500px;" />

    accuracy on test set: 0.9231

#### 🔹 5.1.3) Model #3 (EfficientNetB0)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_freeze.png" style="width:500px;" />

    accuracy on test set: 0.9231

### 5.2) กรณี Feature extractor unfreeze last layer

#### 🔹 5.2.1) Model #1 (VGG16): Feature extractor unfreeze last 2 layers
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_unfreeze_acc.png" style="width:500px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_unfreeze_loss.png" style="width:500px;" />

    accuracy on test set: 0.9179

#### 🔹 5.2.2) Model #2 (ResNet50): Feature extractor unfreeze last 5 layers
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_unfreeze_acc.png" style="width:500px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_unfreeze_loss.png" style="width:500px;" />

    accuracy on test set: 0.9436

#### 🔹 5.2.3) Model #3 (EfficientNetB0): Feature extractor unfreeze last 4 layers
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_unfreeze.png" style="width:500px;" />

    👑 accuracy on test set: 0.9589 👑

🔅 จากผลการทดลองพบว่า Model #3 `EfficientNetB0` กรณี Feature extractor unfreeze last 4 layers ได้ค่า Accuracy test set ดีที่สุดเป็น 0.9589
หลังจากนั้นทำการนำค่า initial random weights ออกเพื่อหาประสิทธิภาพเฉลี่ย ได้ผลตามตารางด้านล่าง 

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/acc_mean.png" style="width:600px;" />

จากตางรางประสิทธิภาพเฉลี่ย สรุปได้ว่า การ Train Model#3 `EfficientNetB0` กรณี Feature extractor unfreeze last 4 layers ได้ค่าเฉลี่ย accuracy เท่ากับ 0.9425 และ SD เท่ากับ 0.0109 

เมื่อนำมาหาค่า F1 – Score ได้ค่าดังนี้ 

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/F1-score.png" style="width:500px;" />

### 🔅 Evaluation metric
ตรวจสอบความแม่นยำของ Model จากการทำ Confusion Matrix ดังตารางด้านล่างนี้ เพื่อดูจำนวนการจำแนกประเภทหางปลากัดที่ถูกต้องและไม่ถูกต้อง 
โดยดูการเปรียบเทียบจาก true class และ predicted class พบว่า class ที่ model จำแนกผิดพลาดมากที่สุด คือ class `double` และ `halfmoon`

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/confuse_matrix1.png" style="width:400px;" />

### 🔅 Visualizing what CNN learned with Grad-Cam

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/gradcam5.png" style="width:400px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/gradcam1.png" style="width:400px;" />          <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/gradcam2.png" style="width:400px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/gradcam3.png" style="width:400px;" />          <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/gradcam4.png" style="width:400px;" />

## 6. Discussion

💡 จากผลการทดลองพบว่า เมื่อเปรียบเทียบ Dataset imagenet กับ Own dataset พบว่าค่า accuracy เป็นไปในทิศทางเดียวกันซึ่งสามารถแสดงให้เห็นดังตารางนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/compare_imagenet.png" style="width:600px;" /> 

💡 จากตาราง Confusion matrix และ F1-Score จะพบว่า classที่1 `double` และ classที่2 `halfmoon` มีโอกาส classify ผิดพลาดสูงกว่า class อื่นๆ ซึ่งแยกผลการ classify ผิดได้ ดังนี้

กรณีที่ 1 ผล classify ผิดจาก `halfmoon` เป็น `double`

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict1-2(1).png" style="width:300px;" /> <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict1-2(2).png" style="width:300px;" />

กรณีที่ 2 ผล classify ผิดจาก `double` เป็น `halfmoon` 

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict2-1(1).png" style="width:300px;" /> <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict2-1(2).png" style="width:300px;" />

จากรูปผลการ classify ที่จำแนกผิดพลาด อาจตั้งสมมติฐานได้ 3 กรณี 
1. ตัวอย่าง x_test ที่ 135 และ 184 พบว่าในส่วนของหางมีสีที่มีเฉดสีแตกต่างกันค่อนข้างมาก
2. ตัวอย่าง x_test ที่ 94 พบว่าลักษณะการว่าย ครีบบนแผ่ไปในระนาบเดียวกับหาง ทำให้ model อาจมองครีบบนเป็นส่วนนึงของหาง
3. ตัวอย่าง x_test ที่ 120 พบว่าลักษณะของมุมกล้องทำให้เห็นครีบหางทั้ง 2 ครีบไม่ชัดเจน

💡 จากกราฟ [`loss validation`](https://github.com/JJMM-DADS/DADS7202_CNN#5-results) จะเห็นว่าเมื่อ train epoch รอบท้ายๆจะพบว่า model มี loss validation สูงขึ้นเรื่อยๆ และ loss train มีค่าค่อนข้างคงที่
แสดงให้เห็นว่าการเพิ่มจำนวน epoch มากจนเกินไป อาจทำให้ model เกิดการ overfit ได้ แต่อย่างไรก็ตาม weight ของ model ที่ใช้สำหรับวัดค่า 
accuracy test set นั้นถูกเก็บไว้ตั้งแต่ epoch ต้นๆแล้ว 

## 7. Conclusion

จากการทดลองเราสามารถจำแนกประเภทของปลากัดตามลักษณะหางทั้ง 4 กลุ่มได้ โดยมีค่า accuracy เฉลี่ยเป็น 94.25% จาก pre-trained model `EfficientNetB0` 
ซึ่งหากพิจารณาจาก ค่า accuracy ของทั้ง 3 models มีความใกล้เคียงกันมาก อยู่ในช่วง 93.075% ± 2.815% แต่หากเทียบกับระยะเวลาของการ train model จะพบว่า model `EfficientNetB0` ใช้ระยะเวลาน้อยกว่า model อื่นถึง 2 เท่า เพราะจำนวน parameter ที่น้อยกว่าอย่างมีนัยสำคัญ

## 8. References

#### Dataset
- https://www.instagram.com/bettaberrythailand/ 
- https://www.secure.instagram.com/betta.corner.th/ 
- https://www.secure.instagram.com/plaricebetta_m/ 
- https://www.secure.instagram.com/nanaabetta/
- https://www.pinterest.com/bettafishcommunity/
- https://www.pinterest.com/nataliamaksimovamoiseeva73/
- https://www.shutterstock.com/search/double-tail-betta
- http://www.lovebettafish.com/category
- https://www.facebook.com/JustBettaSG/photos/?ref=page_internal

#### References 
- [Complete Architectural Details of all EfficientNet Models | by Vardan Agarwal | Towards Data Science](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142)
- https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
- https://iq.opengenus.org/resnet50-architecture/
- [https://iq.opengenus.org/resnet50-architecture/](https://github.com/teehim/BADS7604_hw2)
- https://keras.io/api/applications/


## Members
- (25%) 6410422006 Kan K.   (Prepare dataset + Train and tune EfficientNetB0 model + Summary the report)
- (25%) 6410422010 Chawit T.  (Prepare dataset + Train and tune VGG-16 model + Summary the report)
- (25%) 6410422011 Tharika A. (Prepare dataset + Train and tune ResNet50 model + Summary the report)
- (25%) 6410422028 Nateethon C.     (Prepare dataset + Train and tune VGG-16 model + Summary the report)

#### งานชึ้นนี้เป็นส่วนหนึ่งของรายวิชา Deep Learning (DADS7202) หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์
