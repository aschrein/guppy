# Toy GPU emulator
* Approximates clock cycle cost of instructions
    * ALU latency
    * Sampler/L1/L2/memory latency
* Read only/Write only memory operations
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
mov r1.x, thread_id
mul.u32 r1.x, r1.x, u(4)
ld r2.xy, t0.xw, r1.x
add.u32 r1.y, r1.x, u(4)
ld r3.x, t0.x, r1.y
add.u32 r2.x, r2.x, r2.y
;mov r2.x, r2.x
st u0.x, r1.x, r2.x
ret
```
```assembly
    mov r0.x, lane_id
    mov r1.x, u(0)
    push_mask LOOP_END
LOOP_PROLOG:
    lt.u32 r0.y, r0.x, u(16)
    add.u32 r0.x, r0.x, u(1)
    mask_nz r0.y
LOOP_BEGIN:
    add.u32 r1.x, r1.x, u(1)
    jmp LOOP_PROLOG
LOOP_END:
    ret
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

