## Video timelines for Lesson 5
- [00:00:01](https://youtu.be/J99NV9Cr75I?t=1s) Review of students articles and works
  - "Structured Deep Learning" for structured data using Entity Embeddings,
  - "Fun with small image data-sets (part 2)" with unfreezing layers and downloading images from Google,
  - "How do we train neural networks" technical writing with detailled walk-through,
  - "Plant Seedlings Kaggle competition"
  
- [00:07:45](https://youtu.be/J99NV9Cr75I?t=7m45s) Starting the 2nd half of the course: what's next ?
MovieLens dataset: build an effective collaborative filtering model from scratch

- [00:12:15](https://youtu.be/J99NV9Cr75I?t=12m45s) Why a matrix factorization and not a neural net ?
Using Excel solver for Gradient Descent 'GRG Nonlinear' 

- [00:23:15](https://youtu.be/J99NV9Cr75I?t=23m15s) What are the negative values for 'movieid' & 'userid', and more student questions

- [00:26:00](https://youtu.be/J99NV9Cr75I?t=26m) Collaborative filtering notebook, 'n_factors=', 'CollabFilterDataset.from_csv'

- [00:34:05](https://youtu.be/J99NV9Cr75I?t=34m5s) Dot Product example in PyTorch, module 'DotProduct()'

- [00:41:45](https://youtu.be/J99NV9Cr75I?t=41m45s) Class 'EmbeddingDot()'

- [00:47:05](https://youtu.be/J99NV9Cr75I?t=47m5s) Kaiming He Initialization (via DeepGrid),
sticking an underscore '_' in PyTorch, 'ColumnarModelData.from_data_frame()', 'optim.SGD()'

- Pause

- [00:58:30](https://youtu.be/J99NV9Cr75I?t=58m30s) 'fit()' in 'model.py' walk-through

- [01:00:30](https://youtu.be/J99NV9Cr75I?t=1h30s) Improving the MovieLens model in Excel again,
adding a constant for movies and users called "a bias"

- [01:02:30](https://youtu.be/J99NV9Cr75I?t=1h2m30s) Function 'get_emb(ni, nf)' and Class 'EmbeddingDotBias(nn.Module)', '.squeeze()' for broadcasting in PyTorch

- [01:06:45](https://youtu.be/J99NV9Cr75I?t=1h6m45s) Squeashing the ratings between 1 and 5, with Sigmoid function

- [01:12:30](https://youtu.be/J99NV9Cr75I?t=1h12m30s) What happened in the Netflix prize, looking at 'column_data.py' module and 'get_learner()'

- [01:17:15](https://youtu.be/J99NV9Cr75I?t=1h17m15s) Creating a Neural Net version "of all this", using the 'movielens_emb' tab in our Excel file, the "Mini net" section in 'lesson5-movielens.ipynb'

- [01:33:15](https://youtu.be/J99NV9Cr75I?t=1h33m15s) What is happening inside the "Training Loop", what the optimizer 'optim.SGD()' and 'momentum=' do, spreadsheet 'graddesc.xlsm' basic tab

- [01:41:15](https://youtu.be/J99NV9Cr75I?t=1h41m15s) "You don't need to learn how to calculate derivates & integrals, but you need to learn how to think about the spatially", the 'chain rule', 'jacobian' & 'hessian'

- [01:53:45](https://youtu.be/J99NV9Cr75I?t=1h53m45s) Spreadsheet 'Momentum' tab

- [01:59:05](https://youtu.be/J99NV9Cr75I?t=1h59m5s) Spreasheet 'Adam' tab

- [02:12:01](https://youtu.be/J99NV9Cr75I?t=2h12m1s) Beyond Dropout: 'Weight-decay' or L2 regularization
