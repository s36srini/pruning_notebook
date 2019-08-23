# Pruning Notebook Guide #
This notebook contains both pruning and network surgery for MobileNet V1, but is easily configurable for other models.

## Pruning ##

### Quick Summary ###
Pointwise Convolutional Layers in Mobilenet V1 are pruned by making use of the pruning scheduler that implements a binary mask over a layer's weights, open every ith pruning iteration, the sparsity level K is determined, and based on this sparsity, M channels are masked with zeroes, while the other N - M channels are masked with ones -> this implementation is available in pruning_impl.py in the model-optimization repo.

### User Guide ### 
You must use the pruning docker container which contains the imagenet weights in order to run all the cells in the notebook. If you click on run-all cells, the notebook will automatically run all required cells needed to train Mobilenet V1 from scratch. With 20 epochs, training takes approximately 2.75 days with the Nvidia TITAN X. Once training is complete, a histogram along with the weight summaryof the pruned model will be output, the model can now undergo surgery in order to remove zeroed out channels. 

### Network Surgery ###
Once the model has been pruned, you can run the cell that deletes channels from all pointwise convolutional layers that have a channel sum of zeroa (pointwise conv layer is determined via a regex). Once the new model is output, the model can be saved anywhere. Check the new model's summary to ensure that the weights you intended to remove are in fact removed, and the model has shrunk. 

It is a good idea to retrain this model using the training cell just to ensure that the validation and training accuracy have not changed significantly from the surgery.

To output a frozen model, clear the notebook, run the first cell, then reload the new model that you have saved, and run the cell that is used to output a frozen protobuf.

You can test the viability of your model by making use of the cell that pulls a validation image from the imagenet dataset and outputs the top 5 predictions from the model. Depending on the image, your accuracy could be low since the training mechanism does not apply distortion or cropping.

Congrats, you have just pruned and applied network surgery to a model, making it approximately 60% more compact in size and hopefully faster in inference time after applying quantization! 
