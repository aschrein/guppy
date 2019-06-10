# Toy GPU emulator
## Build
```console
git clone https://github.com/aschrein/guppy && cd guppy
rustup default nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
wasm-pack build
cd www
npm install && npm run start
```
## Profile a test on linux-x86 target
```console
perf record -F99 --call-graph dwarf cargo test mem_test_texture
perf report
```
___
## Features
* Approximates clock cycle cost of instructions
    * ALU latency/pipeline
    * Sampler/L1/L2/memory cache hit/miss latency
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
## Examples of assembly
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
```
```assembly
mov r0.xy, thread_id
and r0.x, r0.x, u(63)
div.u32 r0.y, r0.y, u(64)
mov r0.zw, r0.xy
utof r0.xy, r0.xy
; add 0.5 to fit the center of the texel
add.f32 r0.xy, r0.xy, f2(0.5 0.5)
; normalize coordinates
div.f32 r0.xy, r0.xy, f2(64.0 64.0)
; tx * 2.0 - 1.0
mul.f32 r0.xy, r0.xy, f2(2.0 2.0)
sub.f32 r0.xy, r0.xy, f2(1.0 1.0)
; rotate with pi/4
mul.f32 r4.xy, r0.xy, f2(0.7071 0.7071)
add.f32 r5.x, r4.x, r4.y
sub.f32 r5.y, r4.y, r4.x
;mul.f32 r5.x, r5.x, f(2.0)
mov r0.xy, r5.xy
; texture fetch
; coordinates are normalized
; type conversion and coordinate conversion happens here
sample r1.xyzw, t0.xyzw, s0, r0.xy
; coordinates are u32 here(in texels)
; type conversion needs to happen
st u0.xyzw, r0.zw, r1.xyzw
ret
```
# TODOLIST
* ~~Write tests for RAW, WAW and WAR register hazards, check for antidependency~~
* ~~Add ALU pipelining~~
    * chime variability?
* Register bankning?
* shuffle instructions
* memory io instructions
    * ~~read only raw buffers~~
    * ~~write only raw buffers~~
    * read-write raw buffers
    * ~~write only typed textures~~
    * ~~read only typed textures~~
    * read-write typed textures
    * ~~buffer/texture binding mechanism~~
    * ~~gather/scatter merging~~
    * ~~cache system~~
        * banks
    * ~~sampling system~~
    * atomic operations
        * cache line collision resolution between L1s(false sharing/coalescing)
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

