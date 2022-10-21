# The pre-trained CNN <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/banner-betta-tail.JPG" /> 

## Highlights
- Model `EfficientNetB0` ได้ค่า accuracy เฉลี่ยดีสุดเป็น 94.25% และใช้ระยะเวลาในการ train น้อยกว่า Model `VGG16` และ `ResNet50` ถึง2เท่า
- การเพิ่มจำนวน epoch มากจนเกินไป อาจทำให้ model เกิดการ overfit ได้
- class `double` กับ `halfmoon` รูปภาพมีลักษณะที่คล้ายคลึงกันมาก จนทำให้สี มุมกล้อง และลักษณะการว่ายของปลา ส่งผลให้ model มีการจำแนกผิดพลาดได้

## 1. Introduction

การทดลองนี้จัดทำขึ้นเพื่อออกแบบและสร้าง model ที่สามารถจำแนกประเภทของปลากัดตามลักษณะหาง ซึ่งสามารถช่วยให้ระบุประเภทได้ง่ายขึ้น โดยไม่ต้องอาศัยประสบการณ์ 
หรือมีความรู้เกี่ยวกับสายพันธุ์ปลากัดมาก่อน แบ่งออกเป็น 4 กลุ่ม ได้แก่ `crowntail` `double` `halfmoon` และ `spadetail` 
ซึ่งอาศัยการเก็บรวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันตามกลุ่มที่เราสนใจ มาใช้เป็น input dataset ของ model 
โดยมุ่งเน้นการเปรียบเทียบประสิทธิภาพของ pre-trained model 3 ตัว คือ `VGG16` `ResNet50` และ `EfficientNetB0`


## 2. Data

### 🔹 Data source:
รวบรวมรูปภาพของปลากัดที่มีลักษณะหางแตกต่างกันโดยแบ่งเป็น 4 classes ดังนี้ 

| Class Code No.| Name | No. of image | Image |
| :------: | ------ | ------ | ------ |
| 0 | crowntail | 120 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/crowntail.JPG" style="width:100px;" /> |
| 1 | double | 124 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/double.JPG" style="width:100px;" /> |
| 2 | halfmoon | 121 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/halfmoon.JPG" style="width:100px;" /> |
| 3 | spadetail | 122 | <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/spadetail.JPG" style="width:100px;" /> | 
|   | **Total** | **487**  |   |

Link to download the dataset: https://drive.google.com/drive/folders/17hkb_RNuB67fnempGonKdrPeC84IIjcL?usp=sharing <br />

### 🔹 Data preparation:

####    Data pre-processing & Data Augmentation:

รูปภาพที่เก็บรวมรวบจากหลายแหล่ง และนามสกุลไฟล์ที่แตกต่างจะถูกอ่านด้วย library `opencv` เพื่อให้อยู่ในรูปของ array และ resize เพื่อลดขนาดเป็น<br />
224 x 224 pixel ซึ่งเป็นขนาดมาตรฐานที่ถูกใช้กับ `VGG16` `ResNet50` และ `EfficientNetB0` โดยใช้เทคนิค `INTER_AREA` ซึ่งเป็นการสุ่มตัวอย่างโดยใช้<br />
ความสัมพันธ์เชิงพื้นที่ของพิกเซล หลังจากนั้นเข้าสู่กระบวนการ Data Augmentation โดยกำหนดให้มีการสุ่มดำเนินการ 2 operation คือ 
1. horizontal flip
2. rotation (rotation_range=10)

และสุดท้ายเพื่อให้ Dataset มีการกระจายทุก class ไม่ให้ลำดับติดกันเป็น class เดียวกันทั้งหมด จึงได้คละลำดับของ dataset ใหม่อีกครั้ง และนำไปใช้งานต่อในการ train model
    

### 🔹 Data splitting:

Dataset จะถูก manual split เพื่อแบ่งออกเป็น 3 กลุ่มดังนี้
- train 60%
- validation 20%
- test 20%


## 3. Network architecture

ในส่วนนี้จะให้หลักการ Transfer Learning ด้วย Pre-trained วิธีการนี้เราจะแบ่ง model ออกเป็น 2 ส่วนคือ 
    1. Feature extractor  
    2. Linear Classifier 
โดยส่วน feature extractor จะใช้นำทั้ง architecture และ weighting ของที่ถูก trained มาแล้วมาใช้งาน โดยแบ่งเป็น 3 models ดังนี้

### `VGG16`

จุดเด่นของ VGG16 คือการแทนที่ hyperparameter จำนวนมาก เน้นไปที่การออกแบบ layer conv2D ขนาด 3x3 pixels, 1 stride และการใช้ same padding และ 
max pooling ขนาด 2x2 pixels, 2 stride แบบเดียวกันตลอดทั้งโครงสร้างของสถาปัตยกรรม ในท้ายที่สุดจะมี 2 FC (fully connected layer) 
ตามด้วยการใช้ softmax เป็น Activation function ในการ classify สำหรับ output ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_arch.png" style="width:700px;">

### `ResNet50`

ResNet (Residual Network)เป็นโมเดลที่พัฒนาขึ้นมาเพื่อแก้ปัญหา Vanishing Gradient มีความลึกค่อนข้างมากของ CNN
โดยใช้ Algorithm ที่เรียกว่า Skip Connections ที่สามารถส่งผ่าน Gradient ข้ามไปยัง Layer ต่างๆที่ต้องการได้
ไม่ต้องส่งผ่านแบบ sequential แบบ CNN ซึ่งมี architecture ดังนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet-arch.png" style="width:700px;" />

### `EfficientNetB0`

EfficientNet จะใช้หลักการในการ Scale Model ในทุกๆ dimension ทั้งความ Deep และ Wide ของโมเดลไปพร้อมๆโดยที่จะสามารถแบ่ง model ออกเป็น 7 Block
Block แต่ละตัวมีจำนวน sub-block ที่แตกต่างกันซึ่งมีจำนวนเพิ่มขึ้นตามความDeep และ Wide ของโมเดล โดยเริ่มจาก EfficientNet-B0 จะมีlayer 237 ไปจนถึง EfficientNet-B7 ที่จะมีถึง 813 layer
ซึ่งlayerเหล่านี้จะสร้างมาจาก 5 module ด้านล่าง

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch1.png" style="width:700px;">

module จะถูกรวมเข้าด้วยกันเพื่อสร้าง sub-block ซึ่งจะถูกนำไปใช้เป็นส่วนประกอบในแต่ละ block

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch2.png" style="width:700px;">

โดยทั้งนี้ EfficientNet-B0 จะมี architecture ดังรูปต่อไปนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_arch3.png" style="width:700px;">


ส่วน Linear Classifier มีการกำหนด dense layer มีจำนวน node เป็น 512 และ กำหนด dropout เป็น 0.5 
กำหนด output layer เป็น 4 ใช้ Activation function เป็น softmax


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

ผลการทดลองเปรียบเทียบ Feature extractor แต่ละ model โดยวัดประสิทธิภาพจากค่า accuracy และ ค่า loss แยกผลเป็น 2 กรณี ดังนี้
1. กรณี Freeze all feature extractor

#### Model #1 (VGG16)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_freeze_acc.png" style="width:600px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_freeze_loss.png" style="width:600px;" />
    
    accuracy on test set: 0.9026

#### Model #2 (ResNet50)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_freeze_acc.png" style="width:600px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_freeze_loss.png" style="width:600px;" />

    accuracy on test set: 0.9231

#### Model #3 (EfficientNetB0)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_freeze.png" style="width:600px;" />

    accuracy on test set: 0.9231

2. กรณี Feature extractor unfreeze last layer

#### Model #1 (VGG16)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_unfreeze_acc.png" style="width:600px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/vgg16_unfreeze_loss.png" style="width:600px;" />

    accuracy on test set: 0.9179

#### Model #2 (ResNet50)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_unfreeze_acc.png" style="width:600px;" />
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/resnet_unfreeze_loss.png" style="width:600px;" />

    accuracy on test set: 0.9436

#### Model #3 (EfficientNetB0)
<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/eff_unfreeze.png" style="width:600px;" />

    👑 accuracy on test set: 0.9589 👑

จากผลการทดลองพบว่า Model #3 `EfficientNetB0` กรณี Feature extractor unfreeze last 4 layers ได้ค่า Accuracy test set ดีที่สุดเป็น 0.9589
ทำการนำค่า initial random weights ออกเพื่อหาประสิทธิภาพเฉลี่ยได้ผลตามตารางด้านล่าง 
| No. | 1 | 2 | 3 | 4 | 5 | Mean | SD |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Accuracy | 0.9487 | 0.9333 | 0.9282 | 0.9435 | 0.9589 | 0.9425 | 0.0109 |


เมื่อนำมาหาค่า F1 – Score ได้ค่าดังนี้ 

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/F1_score.png" style="width:400px;" />

Evaluation metric

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/confuse_matrix1.png" style="width:400px;" />

Visualizing what CNN learned with Grad-Cam


## 6. Discussion

จากผลการทดลองพบว่า เมื่อเปรียบเทียบ dataset imagenet กับ own dataset พบว่าค่า accuracy เป็นไปในทิศทางเดียวกันซึ่งสามารถแสดงให้เห็นดังตารางนี้

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/compare_imagenet.png" style="width:500px;" /> 

จากตาราง Confusion matrix และ F1-Score จะพบว่า classที่1 `double` และ classที่2 `halfmoon` มีโอกาสclassifierผิดพลาดสูงกว่า classอื่นๆ ซึ่งเมื่อพิจารณาจากรูป

กรณีที่ 1 ผล classify ผิดจาก `halfmoon` เป็น `double`

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict1-2(1).png" style="width:300px;" /> <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict1-2(2).png" style="width:300px;" />

กรณีที่ 2 ผล classify ผิดจาก `double` เป็น `halfmoon` 

<img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict2-1(1).png" style="width:300px;" /> <img src="https://github.com/JJMM-DADS/DADS7202_CNN/blob/main/images/predict2-1(2).png" style="width:300px;" />

จากรูปผลการ classify ที่จำแนกผิดพลาด อาจตั้งสมมติฐานได้3กรณี 
1.ตัวอย่าง testที่ 135และ184 พบว่าในส่วนของหางมีสีที่มีเฉดสีแตกต่างกันค่อนข้างมาก
2.ตัวอย่าง testที่ 94 พบว่าลักษณะการว่าย ครีบบนแผ่ไปในระนาบเดียวกับหาง ทำให้ modelอาจมองครีบบนเป็นส่วนนึงของหาง
3.ตัวอย่าง testที่ 120 พบว่าลักษณะของมุมกล้องทำให้เห็นครีบหางทั้ง2ครีบไม่ชัดเจน

จากกราฟ loss validation จะเห็นว่าเมื่อ train epoch รอบท้ายๆจะพบว่า model มี loss validation สูงขึ้นเรื่อยๆ และ loss train มีค่าค่อนข้างคงที่
แสดงให้เห็นว่าการเพิ่มจำนวน epoch มากจนเกินไป อาจทำให้ modelเกิดการ overfit ได้ แต่อย่างไรก็ตาม weight ของ model ที่ใช้สำหรับวัดค่า 
acuuracy test set นั้นถูกเก็บไว้ตั้งแต่ epoch ต้นๆแล้ว 

## 7. Conclusion

จากการทดลองเราสามารถจำแนกประเภทของหางปลาทั้ง4กลุ่มได้ โดยมีค่า accuracy เฉลี่ยเป็น 94.25% จาก pre-trained model `EfficientNetB0` หากพิจารณา ค่า accuracy ของทั้ง 3 models 
มีความใกล้เคียงกันมาก อยู่ในช่วง 92% ± 2.5% แต่หากเทียบกับระยะเวลาของการ train model จะพบว่าmodel `EfficientNetB0` ใช้ระยะเวลาน้อยกว่า modelอื่นถึง2เท่า 
เพราะจำนวน parameter ที่น้อยกว่าอย่างมีนัยสำคัญ

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

#### References 
- [Complete Architectural Details of all EfficientNet Models | by Vardan Agarwal | Towards Data Science](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142)
- https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c


## Members
- (25%) 6410422006 กานต์ เกริกชัยวัน   (Prepare dataset + Train and tune EfficientNetB0 model + Summary the report)
- (25%) 6410422010 ชวิศ เตชจินดาวงศ์  (Prepare dataset + Train and tune VGG-16 model + Summary the report)
- (25%) 6410422011 ธาริกา อาจิตรนุภาพ (Prepare dataset + Train and tune ResNet50 model + Summary the report)
- (25%) 6410422028 นทีธร ชุลีกราน     (Prepare dataset + Train and tune VGG-16 model + Summary the report)

#### งานชึ้นนี้เป็นส่วนหนึ่งของรายวิชา Deep Learning (DADS7202) หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์
