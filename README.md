# convolutionalRBM.m

A MATLAB / MEX / CUDA-MEX implementation of Convolutional Restricted Boltzmann Machines.

<span style="color:red;font-weight:bold">Note: This repository is no longer actively maintained, supported, or updated. Use at your own risk.</span>

## References

1. Norouzi, M. [Convolutional Restricted Boltzmann Machines for Feature Learning](https://norouzi.github.io/research/papers/masters_thesis.pdf). 2009.
2. Lee, H., Grosse R., Ranganath, R., and Ng A. Y. [Convolutioanl Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations](http://robotics.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf). ICML 2009.
3. Norouzi, M., Ranjbar, M., and Mori, G. [Stacks of Convolutional Restricted Boltzmann Machines for Shift-Invariant Feature Learning](http://www.cs.toronto.edu/~norouzi/research/papers/conv_rbm.pdf).
IEEE Computer Vision and Pattern Recognition (CVPR), 2009. </li>

## General Information

(From 2014)

When I was doing research on deep learning structures, I was amazed by convolutional RBMs when I read about them because they possess the amazing capability of learning object parts and "recovering" missing parts (see <a href="#reference">Reference</a>). Although <a href="http://www.cs.utoronto.ca/~kriz/" target="_blank">Alex Krizhevsky</a> has an implementation of such network in pure CUDA, I think some (like myself) might want to use a more friendly Matlab version to process smaller amount of data with a bit more time-consumption --- so here we are.</p>

<div class="title"><h3>Status</h3></div>

<p>
<table>
<tr><td><span class="lighter">Current Version</span></td><td>0.3</td></tr>
<tr><td><span class="lighter">Note (Feb. 7, 2014)</span></td><td>New <span class="caps"><span class="caps">CPU</span></span> code updated for new data structure. <span class="caps"><span class="caps">GPU</span></span> code might <emph>not</emph> work. Some updates on help text, code comments, as well as mex option files. New nvmex option file for MacOS added.</td></tr>
<tr><td><span class="lighter">Important Note</span></td><td>This code is <span class="lighter">not</span> an official release by the authors and may contain defect resulting in erroneous outcome. Please use with caution and all possible losses is at the user&#8217;s own risk. For research / personal purposes only and should be kept public with this note. </td></tr>
<tr><td><span class="lighter">Update Log</span></td><td>
<table>
<tr><td><span class="lighter-color">Mar 31, 2014: </span></td><td><emph>Example code added</emph></td></tr>
<tr><td><span class="lighter-color">Feb 7, 2014: 0.3</span></td><td>Major change of data structure, <span class="caps"><span class="caps">CPU</span></span> working version</td></tr>
<tr><td><span class="lighter-color">Nov 28, 2012: 0.2 alpha</span></td><td>Change of file names <span class="amp">&amp;</span> functions</td></tr>
<tr><td><span class="lighter-color">Oct 17, 2012: 0.1 alpha</span></td><td>Basic functionalities</td></tr>
</table>
</td></tr>
</table>
</p>

<div class="title"><h3>Known&nbsp;Issues</h3></div>

<p>
<ul class="news">
<li><span class='lighter'>The <span class="caps"><span class="caps">CUDA</span></span> code might crash or produce rubbish results. </span> Although I make the best effort to keep the <span class="caps"><span class="caps">CPU</span></span> code of this project clean and usable, I sometimes struggle to do the same for the <span class="caps"><span class="caps">CUDA</span></span> code in the same timely manner. If that happens, please accept my deepest apologies and hang tight for my&nbsp;next&nbsp;release.</li>
<li><span class='lighter'>The <span class="caps"><span class="caps">CUDA</span></span> code is currently not running properly (at all) on Mac <span class="caps"><span class="caps">OS</span></span> 10.9.1. </span>This seems to be a general <span class="caps"><span class="caps">CUDA</span></span> issue since the Mavericks update. Please sit tight before nVidia releases new drivers to save&nbsp;the&nbsp;world.</li>
<li><span class='lighter'>The <span class="caps"><span class="caps">CUDA</span></span> code might not compile properly on Linux systems (Ubuntu, etc).</span> I don&#8217;t have a Linux system at hand set up with <span class="caps"><span class="caps">CUDA</span></span> and everything for the moment to test the building scripts. But anyone&#8217;s help on this is very&nbsp;much&nbsp;welcomed.</li>
</ul>
</p>

<div class="title"><h3>FAQs <span class="amp">&amp;</span>&nbsp;Remarks</h3></div>

<p>
<ul class="news">
<li><span class="lighter">I can&#8217;t run the code. Functions not exist?</span><br>To run this code properly, run <span class="code">make</span>, and follow the instructions to setup your compiler for <span class="caps"><span class="caps">MEX</span></span> to compile necessary <span class="caps"><span class="caps">MEX</span></span>&nbsp;files.</li>
<li><span class="lighter">How to use your <span class="caps"><span class="caps">CUDA</span></span>-<span class="caps"><span class="caps">MEX</span></span>?</span><br>See <a href="http://www.cs.ucf.edu/~janaka/gpu/using_nvmex.htm" target="_blank">A Guide to Using <span class="caps"><span class="caps">NVMEX</span></span> Tool</a>. I&#8217;ve modified the nvmexopts.bat file a bit to link the <span class="caps"><span class="caps">CUDA</span></span> libs. <span class="code">make</span> will ask you whether you want to compile <span class="caps"><span class="caps">CUDA</span></span> <span class="caps"><span class="caps">MEX</span></span> files. This feature is limited by the <span class="code">nvmexopts.bat</span> file and currently only work under Win32. Some configuration of the bat file is needed, which is covered in the software documentation&nbsp;(to&nbsp;come).</li>
<li><span class="lighter">How fast can this code run?</span><br>I can&#8217;t say it runs fast: after all the current (0.1 alpha) backbone code is still implemented in Matlab. But with mex implementation, it&#8217;s about 5x faster than a pure Matlab implememtation. However since <span class="caps"><span class="caps">CUDA</span></span>-<span class="caps"><span class="caps">MEX</span></span> is not quite taken advantage of, the current version (0.1 alpha) using <span class="caps"><span class="caps">CUDA</span></span>-<span class="caps"><span class="caps">MEX</span></span> runs about 3x slower on my laptop, than the <span class="caps"><span class="caps">MEX</span></span>&nbsp;version.</li>
<li><span class="lighter">I found a bug. Where do I report to?</span><br>Both commenting below this post and adding an issue at the <a href="https://github.com/qipeng/convolutionalRBM.m" target="_blank">project homepage</a>&nbsp;will&nbsp;do.</li>
<li><span class="lighter">Future development plans?</span><br>This code is still under heavy development before it&#8217;s usable in real-world applications instead of toy-sized problems. A full <span class="caps"><span class="caps">MEX</span></span> / <span class="caps"><span class="caps">CUDA</span></span>-<span class="caps"><span class="caps">MEX</span></span> implementation is planned. Besides, a gadget script for compiling <span class="caps"><span class="caps">MEX</span></span> / <span class="caps"><span class="caps">CUDA</span></span>-<span class="caps"><span class="caps">MEX</span></span> files, as well as sample data and usage,&nbsp;are&nbsp;considered.</li>
</ul>
</p>

## Current status

The current version contains a tested **CPU version only**. The GPU code in this version is incompatible with the CPU implementation, and may result in memory leakage or other issues. The GPU version is still in development. Any updates will be posted here.

CUDA compilation support for multiple platforms has not been thoroughly tested.

For general information on functions in this project, try `help function_name`, as most `.m` files in this project are self-documented.

## Recent FAQ
Please leave questions in Github issues in this repository. Even if I am not around to actively maintain the repository, hopefully the Github community can help you answer your questions more quickly.

That being said, here are some questions from recent comments

* **How do I get the data and parameters for `trainCRBM?`**

You'll have to download a dataset on your own, and transform it to a format that is compatible with `trainCRBM`. For format of the input, try using the MATLAB `help` command on the function.

As for the `param` parameter in that function, try the `getparams` function for an example.

* **What is the `oldModel` parameter in `trainCRBM?`**

Firstly, this parameter is *optional*. That is, you can simply ignore it when you first train your CRBM.

This parameter sort of a fail-safe. As is known to researchers, CDBN's can take a relatively long time to train, and there's usually no guarantee for the system state of your machine running the traning process. Therefore, in my implementation, the model parameters are saved to a `.mat` file periodically during the training process.

In case your training was interrupted accidentally, you can load the intermediate model from the fail-safe `.mat` file, and use the model there as the `oldModel` parameter in this function. The training process will instantly continue based on the saved progress.
