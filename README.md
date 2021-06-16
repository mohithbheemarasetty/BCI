# BCI
An ann algorithm which will detect motor imagery from an eeg device and will give the possible output.

The eeg headset used is from open BCI with 8 channels which are placed on and around the motor cortex. This headset used for this project uses prong electrodes and uses Lab Streaming Layer to transmit data from headset to the device.
The code is universal, so as long as u use LSL to transmit the data the code should work. It is important to make sure the electrodes are placed in the correct positions for most accurate predictions.

Usage of higher channel headsets is recommended as it was hard to differentiate between the data with 8 channels (if using 16 channels or highers, increase the slicing of the data sets accordingly)

The program takes data in a time period of 0.5s and averages them and makes a prediction which is done 10 times (in total takes 5 secs of data) and the prediction with majority will be the output which will be simulated by using the keypress module. The output rate is 1 command per 5 secs which can be changed if needed.
