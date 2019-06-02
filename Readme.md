# TODOLIST
* ~~Write tests for RAW, WAW and WAR register hazards, check for antidependency~~
* Add ALU pipelining
    * chime variability?
* Register bankning?
* shuffle instructions
* memory io instructions
    * ~~read only raw buffers~~
    * ~~write only raw buffers~~
    * write only typed textures
    * read only typed textures
    * ~~buffer/texture binding mechanism~~
    * ~~gather/scatter merging~~
        * ~~cache line collision resolution between L1s(false sharing/coalescing)~~
    * ~~cache system~~
        * banks
    * sampling system
    * atomic operations
* thread group instructions
    * barriers
    * fences
    * SLM
    * atomics
* Refactor
    * Memory subsystem
        * Simplify cache communication
___
# References
Detailed overview of modern GPU architectures
```
A. Bakhoda, G.L. Yuan, W.W.L. Fung, H. Wong, T.M. Aamodt.
"Analyzing CUDA workloads using a detailed GPU simulator,"
Performance Analysis of Systems and Software, 2009. ISPASS 2009.
```
___
Nice overview of reconvergence mechanisms used in various real world hw
```
S. Collange.
"Stack-less SIMT Reconvergence at Low Cost."
TechnicalReport hal-00622654, Universit ́e de Lyon, September 2011.
```
___

Benefits of scalar path
```
Lee,  R.  Krashinsky,  V.  Grover,  S.  W.  Keckler,  and  K.  Asanovic,
“Convergence and scalarization for data-parallel architectures"
2013  IEEE/ACM  InternationalSymposium on, pp. 1–11, 2013.
```
___
Detailed micro-benchmarking of instruction/memory hierarchy for Nvidia hw
```
H. Wong, M.-M. Papadopoulou, M. Sadooghi-Alvandi, andA. Moshovos.
"Demystifying GPU Microarchitecturethrough Microbenchmarking"
InInternational Symposiumon Performance Analysis of Systems and Software, pages235–246, March 2010.
```
___
Compilation of cpu/gpu architecture know hows
```
"Computer Architecture: A Quantitative Approach"
Book by David A Patterson and John L. Hennessy
```
___
Overview of intel integrated GPU architecture(Skylake)
```
Gera, Prasun, et al.
"Performance Characterisation and Simulation of Intel's Integrated GPU Architecture." 2018 I
```
___
Overview of various GPU simulation approaches
```
Kaszyk, Kuba, et al.
"Full-System Simulation of Mobile CPU/GPU Platforms." 2019
```
___
Good blog post about performance implications of swizzle/tiling of textures
```
Fabian Giesen
https://fgiesen.wordpress.com/2011/01/17/texture-tiling-and-swizzling/
```
___

