#convolutionalRBM.m


A MATLAB / MEX / CUDA-MEX implementation of Convolutional Restricted Boltzmann Machines.

## General Information
Please refer to the [project introduction page on my website](http://qipeng.me/software/convolutional-rbm.html).

## Current status (updated: May 10, 2013)
Apparently this project has been dormant for the past a few months, and it will remain dormant for a few months to come, since my timetable is quite full recently. I might probably come back to this project during *June to August*, but I cannot say that for sure.

For general information on functions in this project, try `help function_name`, as most `.m` files in this project are self-documented.

## Some questions raised in recent comments
I'm sincerely sorry if I did not respond in a timely manner on GitHub. In this case, the most efficient way to reach me is via [Email](mailto:qipeng.thu@gmail.com?subject=[CRBM%20Issue]) (Please begin your email subject with "[CRBM Issue]", which should have automatically added if you follow this link). 

Otherwise if you would like to share your questions with others for discussion, try using the Issue feature of GitHub --- I cannot reply to comments.

Here's some questions from recent comments

* **How do I get the data and parameters for `trainCRBM?`**
 
You'll have to download a dataset on your own, and transform it to a format that is compatible with `trainCRBM`. For format of the input, try using the MATLAB help command on the function. 

As for the `param` parameter in that function, try the `getparams` function for an example.

* **What is the oldModel parameter in `trainCRBM?`**

Firstly, this parameter is *optional*. That is, you can simply ignore it when you first train your CRBM.

This parameter sort of a fail-safe. As is known to researchers, CDBN's can take a relatively long time to train, and there's usually no guarantee for the system state of your machine running the traning process. Therefore, in my implementation, the model parameters are saved to a `.mat` file periodically during the training process.

In case your training was interrupted accidentally, you can load the intermediate model from the fail-safe `.mat` file, and use the model there as the `oldModel` parameter in this function. The training process will instantly continue based on the saved progress.