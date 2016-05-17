# Spoofing_Journal_Models

Models for our recently submitted spoofing journal paper on *Speech Communication*: 

>**Deep Features for Automatic Spoofing Detection**

>Yanmin Qian, Nanxin Chen, Kai Yu

We included two best models: DNN model and RNN model.

Dependency:

 * [Keras](https://github.com/fchollet/Keras)
 * [Python script for HTK format](http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/htkmfc.py)
 * [scikit-learn](https://github.com/scikit-learn/scikit-learn) for classification
 * [HTK](http://htk.eng.cam.ac.uk/) for feature extraction

Steps:
 * You should first extract features using *HCopy* and fbank.cfg
 * After that you can try BLSTM model and dnn model. Use *gzip -d* to decompress it.
 * Decoding command: python decode_*.py \<decompressed model_file\> \<scp_file\>

Format for scp file:

>name=file_path([start,end])

>[start,end] is optional

Currently it is under construction.
