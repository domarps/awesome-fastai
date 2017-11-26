## Video timelines for Lesson 2

- [00:01:01](https://youtu.be/JNxcznsrRb8?t=1m1s) Lesson 1 review, image classifier,
PATH structure for training, learning rate,
what are the four columns of numbers in "A Jupyter Widget"

- [00:04:45](https://youtu.be/JNxcznsrRb8?t=4m45s) What is a Learning Rate (LR), LR Finder, mini-batch, 'learn.sched.plot_lr()' & 'learn.sched.plot()', ADAM optimizer intro

- [00:15:00](https://youtu.be/JNxcznsrRb8?t=15m) How to improve your model with more data,
avoid overfitting, use different data augmentation 'aug_tfms='

- [00:18:30](https://youtu.be/JNxcznsrRb8?t=18m30s) More questions on using Learning Rate Finder

- [00:24:10](https://youtu.be/JNxcznsrRb8?t=24m10s) Back to Data Augmentation (DA),
'tfms=' and 'precompute=True', visual examples of Layer detection and activation in pre-trained
networks like ImageNet. Difference between your own computer or AWS, and Crestle.

- [00:29:10](https://youtu.be/JNxcznsrRb8?t=29m10s) Why use 'learn.precompute=False' for Data Augmentation, impact on Accuracy / Train Loss / Validation Loss

- [00:30:15](https://youtu.be/JNxcznsrRb8?t=30m15s) Why use 'cycle_len=1', learning rate annealing,
cosine annealing, Stochastic Gradient Descent (SGD) with Restart approach, Ensemble; "Jeremy's superpower"

- [00:40:35](https://youtu.be/JNxcznsrRb8?t=40m35s) Save your model weights with 'learn.save()' & 'learn.load()', the folders 'tmp' & 'models'

- [00:42:45](https://youtu.be/JNxcznsrRb8?t=42m45s) Question on training a model "from scratch"

- [00:43:45](https://youtu.be/JNxcznsrRb8?t=43m45s) Fine-tuning and differential learning rate,
'learn.unfreeze()', 'lr=np.array()', 'learn.fit(lr, 3, cycle_len=1, cycle_mult=2)'

- [00:55:30](https://youtu.be/JNxcznsrRb8?t=55m30s) Advanced questions: "why do smoother services correlate to more generalized networks ?" and more.

- [01:05:30](https://youtu.be/JNxcznsrRb8?t=1h5m30s) "Is the Fast.ai library used in this course, on top of PyTorch, open-source ?" and why Fast.ai switched from Keras+TensorFlow to PyTorch, creating a high-level library on top.

PAUSE

- [01:11:45](https://youtu.be/JNxcznsrRb8?t=1h11m45s) Classification matrix 'plot_confusion_matrix()'

- [01:13:45](https://youtu.be/JNxcznsrRb8?t=1h13m45s) Easy 8-steps to train a world-class image classifier

- [01:16:30](https://youtu.be/JNxcznsrRb8?t=1h16m30s) New demo with Dog_Breeds_Identification competition on Kaggle, download/import data from Kaggle with 'kaggle-cli', using CSV files with Pandas. 'pd.read_csv()', 'df.pivot_table()', 'val_idxs = get_cv_idxs()'

- [01:29:15](https://youtu.be/JNxcznsrRb8?t=1h29m15s) Dog_Breeds initial model, image_size = 64,
CUDA Out Of Memory (OOM) error

- [01:32:45](https://youtu.be/JNxcznsrRb8?t=1h32m45s) Undocumented Pro-Tip from Jeremy: train on a small size, then use 'learn.set_data()' with a larger data set (like 299 over 224 pixels)

- [01:36:15](https://youtu.be/JNxcznsrRb8?t=1h36m15s) Using Test Time Augmentation ('learn.TTA()') again

- [01:48:10](https://youtu.be/JNxcznsrRb8?t=1h48m10s) How to improve a model/notebook on Dog_Breeds: increase the image size and use a better architecture.
ResneXt (with an X) compared to Resnet. Warning for GPU users: the X version can 2-4 times memory, thus need to reduce Batch_Size to avoid OOM error

- [01:53:00](https://youtu.be/JNxcznsrRb8?t=1h53m) Quick test on Amazon Satellite imagery competition on Kaggle, with multi-labels

- [01:56:30](https://youtu.be/JNxcznsrRb8?t=1h56m30s) Back to your hardware deep learning setup: Crestle vs Paperspace, and AWS who gave approx $200,000 of computing credits to Fast.ai Part1 V2.
More tips on setting up your AWS system as a Fast.ai student, Amazon Machine Image (AMI), 'p2.xlarge',
'aws key pair', 'ssh-keygen', 'id_rsa.pub', 'import key pair', 'git pull', 'conda env update', and how to shut down your $0.90 a minute with 'Instance State => Stop'
