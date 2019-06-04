# Toy GPU emulator
* Approximates clock cycle cost of instructions
    * ALU latency
    * Sampler/L1/L2/memory latency
* Branching
    * Implemented with mask stack
* Loops
    * When all lanes become inactive a mask is popped from the stack
* Architecture inspired by AMD gpu
* Dynamic instruction scheduling
    * WAW, RAW, WAR Hazards are resolved via stalls
    * Stalls happen when a locked register is used, or at the end of wave execution
* DirectX/PTX like assembly syntax
    * Registers have 4 components accessed with swizzle suffix e.g. r0.xyzw
    * Up to 256 registers per lane
    * Register components share lock flag e.g. locking r1.x means r1.y is locked
    * Typed instrucions have type suffix e.g. mul.f32, mul.u32
    * Manual type conversion
```
DRAM -> L2 -+
            |_ (Sampler, L1) - CU scheduler - (wavefront_1..wavefront_N)
            |_ ...
            *
            *
            |_ ...
```
___
## Example
```assembly
; assume group size of 1024
mov r0.xy, thread_id
and r0.x, r0.x, u(0x1f) ; 0b11111
div.u32 r0.y, r0.y, u(32)
mov r0.zw, r0.xy
utof r0.xy, r0.xy
; add 0.5 to fit the center of the texel
add.f32 r0.xy, r0.xy, f2(0.5 0.5)
; r0.xy now is (0.0 .. 31.5, 0.0 .. 31.5)
div.f32 r0.xy, r0.xy, f2(32.0 32.0)
; r0.xy now is (0.0 .. 1.0, 0.0 .. 1.0)
; texture fetch
; coordinates are normalized
; type conversion and coordinate conversion happens here
sample r1.xyzw, t0.xyzw, s0, r0.xy
; coordinates are u32 here(in texels)
; type conversion needs to happen
st u0.xyzw, r0.zw, r1.xy
ret
```
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

