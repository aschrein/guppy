//#![feature(generators, generator_trait)]

struct CacheLine {
    tag: u64,
    data: [u32; 16],
}

struct Cache {
    contents: Vec<CacheLine>,
}

struct Register {
    val: [u32; 4],
    locked: bool,
}

use std::rc::Rc;

type ExecMask = Vec<bool>;
type VGPRF = Vec<Register>;
type SGPRF = Vec<Register>;

struct WaveState {
    vgprfs: Vec<VGPRF>,
    sgprf: SGPRF,
    // id within the same dispatch group
    dispatch_id: u32,
    // Shared pointer to the program
    program: Rc<Program>,
    // Next instruction
    pc: u32,

    exec_mask: ExecMask,
    // execution mask stack
    exec_mask_stack: Vec<ExecMask>,

    // if this wave has been stalled on the previous cycle
    stalled: bool,
    // if this wave executes some job
    enabled: bool,
    // if this wave is being dispatched on this cycle
    has_been_dispatched: bool,
}

struct VALUState {
    instr: Instruction,
    wave_id: u32,
    timer: u32,
    busy: bool,
}

struct SALUState {
    instr: Instruction,
    wave_id: u32,
    timer: u32,
}

struct SamplerState {}

struct L1State {}

struct L2State {}

struct SLMState {}

// Fetch-Execute State
struct FEState {}

struct CUState {
    // ongoing work, some waves might be disabled
    waves: Vec<WaveState>,
    // Vector ALU states
    valus: Vec<VALUState>,
    // Scalar ALU states
    salus: Vec<SALUState>,

    fes: Vec<FEState>,
    // Sampler pipeline states
    samplers: Vec<SamplerState>,
    l1: L1State,
    slm: SLMState,
}

struct GPUState {
    // ongoing work
    cus: Vec<CUState>,
    //
    l2: L2State,
    // queue for future work
    dreqs: Vec<DispatchReq>,
}

struct ALUInstMeta {
    latency: u32,
    throughput: u32,
}

struct GPUConfig {
    // ~300 clk
    DRAM_latency: u32,
    // Bytes per clock: 8, 16, 32, 64, 128, 256, 512
    DRAM_bandwidth: u32,

    // Cache hierarchy config
    // L1 is part of CU
    // ~16kb
    L1_size: u32,
    // ~100 clk
    L1_latency: u32,
    // L2 is shared between CUs
    // ~32kb
    L2_size: u32,
    // ~200 clk
    L2_latency: u32,

    // Sampler config
    // is part of CU
    sampler_cache_size: u32,
    sampler_cache_latency: u32,
    // Number of sample pipelines per CU
    samplers_per_cu: u32,
    // Latency of sampler pipeline w/o memory latency
    sampler_latency: u32,

    // Shared Local Memory is part of CU
    // ~16kb
    SLM_size: u32,
    // ~20clk
    SLM_latency: u32,
    // ~32
    SLM_banks: u32,

    // Number of [u32 X 4] per Processing Unit
    VGPRF_per_pe: u32,
    // Number of [u32 X 4] per CU
    SGPRF_per_wave: u32,

    // SIMD width of ALU
    wave_size: u32,
    // Compute Unit count per GPU
    CU_count: u32,
    ALU_per_cu: u32,
    // Number of SIMD threads which execution is handled by CU
    waves_per_cu: u32,
    // Number of fetch decode units per CU
    fd_per_cu: u32,
}

#[derive(Debug, Clone)]
enum Component {
    X,
    Y,
    Z,
    W,
    NONE,
}

struct Buffer {
    raw_data: Vec<u8>,
    size: u32,
}

struct Texture {
    raw_data: Vec<u8>,
    pitch: u32,
    height: u32,
}

enum WrapMode {
    CLAMP,
    WRAP,
    BLACK,
}

enum SampleMode {
    POINT,
    BILINEAR,
}

enum SampleFormat {
    RGBA8_NORMAL,
    RGB8_NORMAL,
}

struct Sampler {
    wrap_mode: WrapMode,
    sample_mode: SampleMode,
    sample_format: SampleFormat,
}

#[derive(Debug, Clone)]
enum Interpretation {
    F32,
    I32,
    U32,
    NONE,
}

// r1.__xy
#[derive(Debug, Clone)]
struct RegRef {
    id: u32,
    comps: [Component; 4],
    interp: Interpretation,
}

#[derive(Debug, Clone)]
struct BufferRef {
    id: u32,
}

#[derive(Debug, Clone)]
struct TextureRef {
    id: u32,
}

#[derive(Debug, Clone)]
enum ImmediateVal {
    V1U(u32),
    V2U(u32, u32),
    V3U(u32, u32, u32),
    V4U(u32, u32, u32, u32),
    V1S(i32),
    V2S(i32, i32),
    V3S(i32, i32, i32),
    V4S(i32, i32, i32, i32),
    V1F(f32),
    V2F(f32, f32),
    V3F(f32, f32, f32),
    V4F(f32, f32, f32, f32),
}

#[derive(Debug, Clone)]
enum BuiltinVal {
    // Global thread id
    THREAD_ID,
    WAVE_ID,
    // ID within wave
    LANE_ID,
}

#[derive(Debug, Clone)]
enum Operand {
    VRegister(RegRef),
    SRegister(RegRef),
    Buffer(BufferRef),
    Texture(TextureRef),
    Immediate(ImmediateVal),
    Builtin(BuiltinVal),
    NONE,
}

#[derive(Debug, Clone)]
enum InstTy {
    MOV,
    ADD,
    SUB,
    MUL,
    DIV,
    NORM,
    SQRT,
    FSQRT,
    LD,
    ST,
    SAMPLE,
    DISCARD,
    NONE,
}

#[derive(Debug, Clone)]
struct Instruction {
    ty: InstTy,
    line: u32,
    ops: [Operand; 4],
}

impl Instruction {
    fn assertTwoOp(&self) {
        match (&self.ops[2], &self.ops[3]) {
            (Operand::NONE, Operand::NONE) => {}
            _ => {
                std::panic!("");
            }
        }
    }
    fn assertThreeOp(&self) {
        match (&self.ops[3]) {
            Operand::NONE => {}
            _ => {
                std::panic!("");
            }
        }
    }
}

//use std::ops::Generator;

enum CmdStatus {
    YIELD,
    COMPLETE,
}

// Generator function for an instruction
type InstGen = Box<Fn(&mut WaveState) -> CmdStatus>;

trait ICmd {
    fn prereq(&mut self, inst: &Instruction) -> bool;
    fn start(&mut self) -> InstGen;
}

fn matchInst(inst: &Instruction) -> Option<InstGen> {
    None
}

struct Program {
    ins: Vec<Instruction>,
}

struct DispatchReq {
    program: Rc<Program>,
    count: u32,
}

fn dispatch(gpu_state: &mut GPUState, program: Program, count: u32) {
    let disp_req = DispatchReq {
        program: Rc::new(program),
        count: count,
    };
    gpu_state.dreqs.push(disp_req);
}

fn clock(gpu_state: &mut GPUState) {
    // @Dispatch work
    {
        let cnt_free_waves = {
            let mut cnt = 0;
            for cu in &gpu_state.cus {
                for wave in &cu.waves {
                    if !wave.enabled {
                        cnt = cnt + 1;
                    }
                }
            }
            cnt
        };
        for i in 0..cnt_free_waves {
            match gpu_state.dreqs.pop() {
                Some(req) => {
                    // Dispatch on this wave
                }
                None => break,
            }
        }
    }
    // @Fetch-Submit part
    {
        // For each compute unit(they run in parallel in our imagination)
        for cu in &mut gpu_state.cus {
            // Clear some flags
            for wave in &mut cu.waves {
                wave.has_been_dispatched = false;
                //wave.stalled = false;
            }
            // For each fetch-exec unit
            for fe in &mut cu.fes {
                for wave in &mut cu.waves {
                    if wave.enabled && !wave.has_been_dispatched && !wave.stalled {
                        let mut dispatchOnVALU = false;
                        let mut dispatchOnSALU = false;
                        let mut dispatchOnSampler = false;
                        let inst = &(*wave.program).ins[wave.pc as usize];
                        let mut has_dep = false;
                        // Do simple decoding and sanity checks
                        match &inst.ty {
                            ADD => {
                                inst.assertThreeOp();

                                for op in &inst.ops {
                                    match &op {
                                        Operand::VRegister(vreg) => {
                                            for gprf in &wave.vgprfs {
                                                if gprf[vreg.id as usize].locked {
                                                    has_dep = true;
                                                }
                                            }
                                            dispatchOnVALU = true;
                                        }
                                        Operand::SRegister(sreg) => {
                                            if wave.sgprf[sreg.id as usize].locked {
                                                has_dep = true;
                                            }

                                            dispatchOnSALU = true;
                                        }
                                        Operand::Immediate(imm) => {}
                                        _ => {
                                            std::panic!("");
                                        }
                                    }
                                }
                            }
                            _ => {
                                std::panic!("");
                            }
                        };
                        // Make sure only one dispatch flag is set
                        assert!(
                            vec![dispatchOnSALU, dispatchOnSampler, dispatchOnVALU]
                                .iter()
                                .filter(|&a| *a == true)
                                .collect::<Vec<&bool>>()
                                .len()
                                == 1
                        );
                        // One of the operands is being locked
                        // Stall the wave
                        if has_dep {
                            wave.stalled = true;
                            continue;
                        }
                        if dispatchOnVALU {
                            for valu in &mut cu.valus {
                                if valu.busy {
                                    continue;
                                }
                                wave.has_been_dispatched = true;
                                valu.instr = inst.clone();
                                valu.timer = 1;
                                valu.busy = true;
                                valu.wave_id = wave.dispatch_id;
                            }
                        } else if dispatchOnSALU {
                            std::panic!();
                        } else if dispatchOnSampler {
                            std::panic!();
                        }
                    }
                }
            }
            // @ALU
            // Now do work on Vector ALUs
            for valu in &mut cu.valus {
                if !valu.busy {
                    continue;
                }
                if valu.timer != 0 {
                    valu.timer -= 1;
                }
                if valu.timer == 0 {
                    let inst = &valu.instr;
                    match &inst {
                        ADD => match (&inst.ops[0], &inst.ops[1]) {
                            (Operand::VRegister(op1), Operand::VRegister(op2)) => {}

                            _ => std::panic!(""),
                        },
                        _ => std::panic!("unsupported {:?}", inst.ops[0]),
                    }
                }
            }
            // And on Scalar ALUs
            for salu in &mut cu.salus {}
        }
    }
}

fn parse(text: &str) -> Vec<Instruction> {
    vec![]
}

fn main() {
    println!("Hello, world!");
}
