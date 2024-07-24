# An-Open-source-HLS-Fully-Parameterizable-Matrix-Multiplication-Library-for-AMD-FPGAs

We present a generic open-source solution for HLS design flows allowing for the implementation of high-performance dense matrix multiplication on modern AMD FPGAs. 

##Specifically, the contribution of this paper can be summarized as follows:
-	An open-source library  designed to accelerate dense matrix multiplication of any size and datatype by extracting parallelism in two dimensions. This purely Synthesizable C library provides very high configurability and flexibility in order to take full advantage of the resources of modern AMD FPGAs without any dependency on the hardware implementation tool (e.g. version of the tools) or any external library.
-	An innovative flow that enables designers to easily develop FPGA-accelerated applications, that involve matrix handling, using the presented fundamental structures minimizing the development and verification time while achieving high performance and energy efficiency.
-	The effectiveness and performance of this library have been rigorously and comprehensively evaluated when implemented on both small and high-end FPGAs and it has been proved that our solution outperforms CPUs, GPUs and even relevant FPGA-tailored approaches.

## References
<a id="1">[1]</a> 
 A. Athanasiadis, N. Tampouratzis and I. Papaefstathiou,  “An Open-source HLS Fully Parameterizable Matrix Multiplication Library for AMD FPGAs - WiP ”
