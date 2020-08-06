# BCI
A machine learning algorithm which will detect imagined motion from an eeg device and will give the possible output.
The imaginary.csv file has the training data for motor imagery and the test.csv has the test data we check the accuracy on.

By using socket we send the predicted result from the data to another device i.e. client to simulate the key press on the device.

Now we can give inputs to the device remotely with just the data from motor imagery which is collected from the bci device.
