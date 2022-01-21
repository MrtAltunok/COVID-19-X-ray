# COVID-19-X-ray
Automated detection of COVID-19 cases using deep neural networks with X-ray images
</br></br>

Dataset, code and trained model <br> https://www.kaggle.com/muraterzurum/covid-19-prossesing<br>

# Purpose of the study</br>
</br>

In December 2019, we encountered the Corona virus disease (COVID -19), which has no place in our lives. This encounter caused great harm to human health and peace on a global scale. As of today, it continues to spread worldwide with more than 260 million confirmed cases and more than 5 million deaths [1]. Early diagnosis plays a major role in reducing the spread. Making the necessary diagnoses also depends on the correct tests. Inadequacy of the hospitals in terms of testing facilities or the test result not being available in the desired time affects the course of the spread. Therefore, there is a need for automatic diagnosis systems that can provide fast results and reduce the diagnostic error to a great extent.
<br></br>
In our study, it is aimed to train a Deep Learning model, which is trained using chest x-rays of healthy people and people with COVID-19 disease for early diagnosis and accurate diagnosis. After this training, we want to be able to distinguish the newly given chest x-ray images as positive or negative.<br>
## Data Set Used
</br>
Chest X-rays of healthy people and people with COVID-19 disease will be used as a data set.<br>
A total of 6259 images will be used for the training. 2975 of these images are healthy, 3284 are images of Covid disease.</br>
A total of 177 images will be used for test. 65 of these images are healthy, 112 of them are Covid images and the number of them is Covid disease images.</br>
A total of 491 images will be used for the validation. 271 of these images are healthy and 220 are images of Covid disease.</br>

If we examine the performance values after the training: <br>
- Model accuracy  0.95<br>
- Model recall 0.91<br>
- The precision of the model is 0.97<br>
- Model's score is 0.94<br>

<img src="https://raw.githubusercontent.com/MrtAltunok/COVID-19-X-ray/main/Resim1.png" alt="Resim1.png">
<img src="https://raw.githubusercontent.com/MrtAltunok/COVID-19-X-ray/main/Resim2.png" alt="Resim2.png">
