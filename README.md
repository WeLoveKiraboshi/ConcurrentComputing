# GPU performance Test with CUDA
**Authors:** [Yuki Saito], [Amar Alkadic] 

This is a repository of JEMARO special lecture: comcurrent computing for Robotics by Dr.Jasmin Jahic.

<a href="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/JEMARO_Logo.png" target="_blank"><img src="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/JEMARO_Logo.png" 
alt="JEMAROLogo" width="584" height="413" border="300" /></a>



### Publications and Report:

You can download our presentation file & essay files below. All rights are reserved in Yuki Saito, and Amar Alkhadic.

GPUs and Parallelism [**[Presentation1](http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/Presentation_Group_1_Assignment_1_GPUsAndPrarellism_compressed.pdf)**]

CUDA for parall computing [**[Presentation2](http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/Presentation_Group_1_Assignment_2_CudaForPrarellComputation_compressed.pdf)**]

GPUs hardware architecture & suitable problems [**[Essay1](http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/Essay_Group_1_Assignment_1_Yuki_Saito_GPUarchitectureAndPrarellism_compressed.pdf)**]

CUDA software architecture & speed-up perfomance test [**[Essay2](http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/Essay_Group_1_Assignment_2_Yuki_Saito_ExperimentsWithCUDA_compressed.pdf)**]


# 1. structure
```
FOLER/
  C++_CUDA/
      image_filtering/
      array_add/
      matrix_mul/
```

#2. run the model
To test our scripts, please run Makefile with
```
make 
```
For matrix_mul, you can get execution file
```
./a.out
```
For array_add, 
```
./add
```
For image_filtering
```
./build/filters_gpu
```

#3. Results

### Matrix multiplication Results (latency & gain)

<a href="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/ArrayMultiplicationResult.png" target="_blank"><img src="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/ArrayMultiplicationResult.png"
alt="JEMAROLogo" width="584" height="413" border="300" /></a>

### Bilateral Image Filtering Results (latency & gain)

<a href="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/BilteralFilteringResult.png" target="_blank"><img src="http://www.hvrl.ics.keio.ac.jp/saito_y/images/JEMARO/ConcurrentComputing/BilteralFilteringResult.png"
alt="JEMAROLogo" width="584" height="413" border="300" /></a>


# 4. Demo Results

now creating...

[comment]: <> (<a href="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png/" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png")

[comment]: <> (alt="ORB-SLAM2" width="916" height="197" border="30" /></a>)
