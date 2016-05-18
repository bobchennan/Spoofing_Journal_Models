# Spoofing_Journal_Models

Models for our recently submitted spoofing journal paper on *Speech Communication*: 

>**Deep Features for Automatic Spoofing Detection**

>Yanmin Qian, Nanxin Chen, Kai Yu

We included two best models: DNN model and RNN model.

Dependency:

 * [Keras==0.3.3](https://github.com/fchollet/Keras)
 * [Python script for HTK format](http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/htkmfc.py)
 * [scikit-learn](https://github.com/scikit-learn/scikit-learn) for classification
 * [HTK](http://htk.eng.cam.ac.uk/) for feature extraction

Steps:
 * You should first extract features using *HCopy* and fbank.cfg
 * After that you can try BLSTM model and dnn model. Use *gzip -d* to decompress it.
 * Decoding command: python decode_*.py \<decompressed model_file\> \<scp_file\>
 * Classify command: python \<classifier.py\> \<training_set\> \<cv_set\> \<development_set\> \<test_set\>

Format for scp file:

>name=file_path([start,end])

>[start,end] is optional

Format for feature file using for different classifiers(at least required for training set):

>label x\_1 x\_2 ... x\_n

## Results

|     Model     |   Classifier  |  S1  |  S2  |  S3  |  S4  |  S5  |  S6  |  S7  |  S8  |  S9  |  S10 | known | unknown | all |
| ------------- | ------------- |------|------|------|------|------|------|------|------|------|------|-------|---------|-----|
|      DNN      |      LDA      |0.03%|0.08%|0.00%|0.00%|0.16%|0.18%|0.02%|0.01%|0.04%|25.47%|0.05%|5.14%|2.60%|
|      DNN      |      GDF      |9.50%|28.19%|4.57%|4.36%|36.59%|32.66%|32.44%|7.62%|30.80%|39.97%|16.64%|28.70%|22.67%|
|      DNN      |      SVM      |0.20%|0.19%|0.01%|0.03%|0.67%|0.65%|0.09%|0.01%|0.15%|37.12%|0.22%|7.60%|3.91%|
|      DNN      |  OneClassSVM  |0.85%|1.03%|0.03%|0.03%|3.69%|3.27%|0.74%|0.01%|0.55%|48.79%|1.12%|10.67%|5.90%|
|     BLSTM     |      LDA      |0.01%|0.24%|0.00%|0.00%|0.12%|0.28%|0.17%|0.03%|0.18%|15.28%|0.07%|3.19%|1.63%|
|     BLSTM     |      GDF      |0.01%|0.66%|0.00%|0.00%|0.15%|0.51%|0.58%|0.09%|0.52%|19.56%|0.16%|4.25%|2.21%|
|     BLSTM     |      SVM      |0.01%|0.85%|0.00%|0.00%|0.26%|0.80%|0.46%|0.03%|0.66%|10.72%|0.22%|2.54%|1.38%|
|     BLSTM     |  OneClassSVM  |0.01%|0.68%|0.00%|0.00%|0.18%|0.59%|0.47%|0.07%|0.50%|11.53%|0.17%|2.63%|1.40%|

Currently it is under construction.
