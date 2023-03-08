# Names classification with RNNs
This repository provides a basic implementation of how to use Recurrent Neural Networks in Pytorch with a case study on predicting the nationality of people from their names 
This repository is inspired by [this tutorial](https://www.youtube.com/watch?v=WEV61GmmPrk&t=1526s).
The basic code described in that tutorial has been extended in this repo by:
- Including support for multiple types of RNN
- Allowing for learning in batches
- Including possibility for padding of sequences (same padding for all sequences or padding of different length per batch)
- Code prettified

Data can be downloaded from [here](https://download.pytorch.org/tutorial/data.zip) and placed inside the `data` folder