//#![feature(generators, generator_trait)]

struct CacheLine {
    tag: u64,
    data: [u32; 16],
}

struct Cache {
    contents: Vec<CacheLine>,
}

type Value = [u32; 4];
type FValue = [f32; 4];

#[derive(Clone, Copy, Debug)]
struct Register {
    val: Value,
    locked: bool,
}

fn AddU32(this: &Value, that: &Value) -> Value {
    [
        this[0] + that[0],
        this[1] + that[1],
        this[2] + that[2],
        this[3] + that[3],
    ]
}

fn AddF32(this_: &Value, that_: &Value) -> Value {
    let this = castToFValue(this_);
    let that = castToFValue(that_);
    castToValue(&[
        this[0] + that[0],
        this[1] + that[1],
        this[2] + that[2],
        this[3] + that[3],
    ])
}

fn castToValue(fval: &FValue) -> Value {
    unsafe {
        let ux = std::mem::transmute_copy(&fval[0]);
        let uy = std::mem::transmute_copy(&fval[1]);
        let uz = std::mem::transmute_copy(&fval[2]);
        let uw = std::mem::transmute_copy(&fval[3]);
        [ux, uy, uz, uw]
    }
}

fn castToFValue(fval: &Value) -> FValue {
    unsafe {
        let ux = std::mem::transmute_copy(&fval[0]);
        let uy = std::mem::transmute_copy(&fval[1]);
        let uz = std::mem::transmute_copy(&fval[2]);
        let uw = std::mem::transmute_copy(&fval[3]);
        [ux, uy, uz, uw]
    }
}

use std::rc::Rc;

type ExecMask = Vec<bool>;
type VGPRF = Vec<Register>;
type SGPRF = Vec<Register>;

struct WaveState {
    // Array of vector registers, vgprfs[0] means array of r0 for each laneS
    vgprfs: Vec<VGPRF>,
    sgprf: SGPRF,
    // id within the same dispatch group
    dispatch_id: u32,
    // Shared pointer to the program
    program: Option<Rc<Program>>,
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

impl WaveState {
    fn print(&self) {
        println!("WaveState:{}", self.dispatch_id);
        for vreg in &self.vgprfs {
            println!("{:?}", vreg
            .iter()
            .map(|x| castToFValue(&x.val))
            .collect::<Vec<FValue>>()
            );
        }
    }
    fn new(config: &GPUConfig) -> WaveState {
        WaveState {
            vgprfs: Vec::new(),
            sgprf: Vec::new(),
            dispatch_id: 0,
            program: None,
            pc: 0,
            exec_mask: Vec::new(),
            exec_mask_stack: Vec::new(),
            stalled: false,
            enabled: false,
            has_been_dispatched: false,
        }
    }
    fn dispatch(&mut self, config: &GPUConfig, program: &Rc<Program>) {
        let mut vgprfs: Vec<VGPRF> = Vec::new();
        let mut sgprf: SGPRF = Vec::new();
        for i in 0..config.VGPRF_per_pe {
            let mut vpgrf: VGPRF = Vec::new();
            for i in 0..config.wave_size {
                vpgrf.push(Register {
                    val: [0, 0, 0, 0],
                    locked: false,
                });
            }
            vgprfs.push(vpgrf);
        }
        for i in 0..config.SGPRF_per_wave {
            sgprf.push(Register {
                val: [0, 0, 0, 0],
                locked: false,
            });
        }
        let mut exec_mask: Vec<bool> = Vec::new();
        for i in 0..config.wave_size {
            exec_mask.push(true);
        }
        self.program = Some(program.clone());
        self.sgprf = sgprf;
        self.vgprfs = vgprfs;
        self.exec_mask = exec_mask;
        self.stalled = false;
        self.pc = 0;
        // @TODO: make something meaningful with the id
        self.dispatch_id = 0;
        self.has_been_dispatched = false;
        self.enabled = true;
    }
}

struct VALUState {
    instr: Option<Instruction>,
    wave_id: u32,
    timer: u32,
    busy: bool,
}

struct SALUState {
    instr: Option<Instruction>,
    wave_id: u32,
    timer: u32,
    busy: bool,
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

impl CUState {
    fn new(config: &GPUConfig) -> CUState {
        let mut waves: Vec<WaveState> = Vec::new();
        for i in 0..config.waves_per_cu {
            waves.push(WaveState::new(config));
        }
        let mut valus: Vec<VALUState> = Vec::new();
        let mut salus: Vec<SALUState> = Vec::new();
        for i in 0..config.ALU_per_cu {
            valus.push(VALUState {
                instr: None,
                wave_id: 0,
                timer: 0,
                busy: false,
            });
        }
        salus.push(SALUState {
            instr: None,
            wave_id: 0,
            timer: 0,
            busy: false,
        });
        let mut fes: Vec<FEState> = Vec::new();
        for i in 0..config.fd_per_cu {
            fes.push(FEState {});
        }
        let mut samplers: Vec<SamplerState> = Vec::new();
        for i in 0..config.samplers_per_cu {
            samplers.push(SamplerState {});
        }
        CUState {
            waves: waves,
            valus: valus,
            salus: salus,
            fes: fes,
            samplers: samplers,
            l1: L1State {},
            slm: SLMState {},
        }
    }
}

struct GPUState {
    // ongoing work
    cus: Vec<CUState>,
    //
    l2: L2State,
    // queue for future work
    dreqs: Vec<DispatchReq>,
    // factory settings
    config: GPUConfig,
}

impl GPUState {
    fn new(config: &GPUConfig) -> GPUState {
        let mut cus: Vec<CUState> = Vec::new();
        for i in 0..config.CU_count {
            cus.push(CUState::new(config));
        }
        GPUState {
            cus: cus,
            l2: L2State {},
            dreqs: Vec::new(),
            config: config.clone(),
        }
    }
}

struct ALUInstMeta {
    latency: u32,
    throughput: u32,
}

#[derive(Clone, Copy)]
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
    //interp: Interpretation,
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
    group_size: u32,
    group_count: u32,
}

fn dispatch(gpu_state: &mut GPUState, program: Program, group_size: u32, group_count: u32) {
    let disp_req = DispatchReq {
        program: Rc::new(program),
        group_count: group_count,
        group_size: group_size,
    };
    gpu_state.dreqs.push(disp_req);
}

fn clock(gpu_state: &mut GPUState) {
    // @Dispatch work
    {
        // @TODO: Remove
        let mut cnt_free_waves = {
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
        let mut deferredReq: Vec<DispatchReq> = Vec::new();
        while cnt_free_waves > 0 {
            match gpu_state.dreqs.pop() {
                Some(req) => {
                    // Make sure it's a power of two and no greater than 1024
                    assert!(
                        ((req.group_size - 1) & req.group_size) == 0
                            && req.group_size <= 1024
                            && req.group_count * req.group_size != 0
                    );
                    assert!(req.group_count * req.group_size % 32 == 0);
                    let mut warp_count = req.group_count * req.group_size / 32;
                    if warp_count > cnt_free_waves {
                        deferredReq.push(req);
                        continue;
                    }
                    cnt_free_waves -= warp_count;
                    // Find a free wave and dispatch on it
                    // @TODO: make popWave method
                    for cu in &mut gpu_state.cus {
                        for wave in &mut cu.waves {
                            if !wave.enabled {
                                wave.dispatch(&gpu_state.config, &req.program);
                                warp_count -= 1;
                                if warp_count == 0 {
                                    break;
                                }
                            }
                        }
                        if warp_count == 0 {
                            break;
                        }
                    }
                }
                None => break,
            }
        }
        gpu_state.dreqs.append(&mut deferredReq);
    }

    // @Fetch-Submit part
    {
        // For each compute unit(they run in parallel in our imagination)
        for cu in &mut gpu_state.cus {
            // Clear some flags
            for wave in &mut cu.waves {
                wave.has_been_dispatched = false;
                wave.stalled = false;
            }
            // For each fetch-exec unit
            for fe in &mut cu.fes {
                for (wave_id, wave) in &mut cu.waves.iter_mut().enumerate() {
                    if wave.enabled && !wave.has_been_dispatched && !wave.stalled {
                        let mut hasVRegOps = false;
                        let mut hasSRegOps = false;
                        let mut dispatchOnSampler = false;
                        let inst = &(*wave.program.as_ref().unwrap()).ins[wave.pc as usize];
                        let mut has_dep = false;
                        // Do simple decoding and sanity checks
                        for op in &inst.ops {
                            match &op {
                                Operand::VRegister(vreg) => {
                                    for reg in &wave.vgprfs[vreg.id as usize] {
                                        if reg.locked {
                                            has_dep = true;
                                        }
                                    }
                                    hasVRegOps = true;
                                }
                                Operand::SRegister(sreg) => {
                                    if wave.sgprf[sreg.id as usize].locked {
                                        has_dep = true;
                                    }

                                    hasSRegOps = true;
                                }
                                Operand::Immediate(imm) => {}
                                Operand::NONE => {}
                                _ => {
                                    std::panic!("");
                                }
                            }
                        }
                        match &inst.ty {
                            InstTy::ADD | InstTy::SUB | InstTy::MUL | InstTy::DIV => {
                                inst.assertThreeOp();
                            }
                            InstTy::MOV => {
                                inst.assertTwoOp();
                            }
                            _ => {
                                std::panic!("");
                            }
                        };
                        //
                        assert!(
                            vec![hasSRegOps, dispatchOnSampler, hasVRegOps]
                                .iter()
                                .filter(|&a| *a == true)
                                .collect::<Vec<&bool>>()
                                .len()
                                == 1 // Mixing different register types
                        );

                        // One of the operands is being locked
                        // Stall the wave
                        if has_dep {
                            wave.stalled = true;
                            continue;
                        }
                        // For simplicity dispatch commands basing only on registers used
                        if hasVRegOps {
                            for valu in &mut cu.valus {
                                if valu.busy {
                                    continue;
                                }
                                wave.has_been_dispatched = true;
                                valu.instr = Some(inst.clone());
                                // @TODO: determine the timer value
                                valu.timer = 1;
                                valu.busy = true;
                                valu.wave_id = wave_id as u32;
                                // Lock the destination registers
                                match &inst.ty {
                                    InstTy::ADD | InstTy::SUB | InstTy::MUL | InstTy::DIV => {
                                        if let Operand::VRegister(dst) = &inst.ops[0] {
                                            for item in &mut wave.vgprfs[dst.id as usize] {
                                                item.locked = true;
                                            }
                                        } else {
                                            std::panic!("")
                                        }
                                    }
                                    InstTy::MOV => {
                                        if let Operand::VRegister(dst) = &inst.ops[0] {
                                            for item in &mut wave.vgprfs[dst.id as usize] {
                                                item.locked = true;
                                            }
                                        } else {
                                            std::panic!("")
                                        }
                                    }
                                    _ => std::panic!(""),
                                }
                                // Successfully dispatched
                                break;
                            }
                        } else if hasSRegOps {
                            std::panic!();
                        } else if dispatchOnSampler {
                            std::panic!();
                        }
                        // If an instruction was issued then increment PC
                        if wave.has_been_dispatched {
                            wave.pc += 1;
                            if wave.pc as usize == wave.program.as_ref().unwrap().ins.len() {
                                wave.enabled = false;
                                // Wave has been retired
                            }
                        } else {
                            // Not enough resources to dispatch a command
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
                    let inst = &valu.instr.as_ref().unwrap();
                    let mut wave = &mut cu.waves[valu.wave_id as usize];
                    let getVal = |wave: &WaveState, op: &Operand| -> Vec<Value> {
                        match &op {
                            Operand::VRegister(vop) => wave.vgprfs[vop.id as usize]
                                .iter()
                                .map(|r| r.val)
                                .collect::<Vec<Value>>(),
                            Operand::Immediate(imm) => match imm {
                                ImmediateVal::V4F(x, y, z, w) => {
                                    let mut values: Vec<Value> = Vec::new();
                                    for i in 0..wave.vgprfs[0].len() {
                                        values.push(castToValue(&[*x, *y, *z, *w]));
                                    }
                                    values
                                }
                                _ => std::panic!(""),
                            },
                            // @TODO: immediate
                            _ => std::panic!(""),
                        }
                    };
                    match &inst.ty {
                        InstTy::ADD | InstTy::SUB | InstTy::MUL | InstTy::DIV => {
                            let src1 = getVal(wave, &inst.ops[1]);
                            let src2 = getVal(wave, &inst.ops[2]);
                            let result = match &inst.ty {
                                // @TODO: sub, div, mul
                                InstTy::ADD => {
                                    src1.iter()
                                        .zip(src2.iter())
                                        // @TODO: Support different types
                                        .map(|(&x1, &x2)| Register {
                                            val: AddF32(&x1, &x2),
                                            locked: false,
                                        })
                                        .collect::<Vec<Register>>()
                                }
                                _ => std::panic!(""),
                            };
                            let dst = match &inst.ops[0] {
                                Operand::VRegister(dst) => dst,
                                _ => std::panic!(""),
                            };
                            assert!(wave.vgprfs[dst.id as usize].len() == result.len());
                            // Registers should be unlocked by this time
                            wave.vgprfs[dst.id as usize] = result;
                        }
                        InstTy::MOV => {
                            let src1 = getVal(wave, &inst.ops[1]);
                            let result = src1
                                .iter()
                                .map(|(&x1)| Register {
                                    val: x1,
                                    locked: false,
                                })
                                .collect::<Vec<Register>>();

                            let dst = match &inst.ops[0] {
                                Operand::VRegister(dst) => dst,
                                _ => std::panic!(""),
                            };
                            assert!(wave.vgprfs[dst.id as usize].len() == result.len());
                            // Registers should be unlocked by this time
                            wave.vgprfs[dst.id as usize] = result;
                        }
                        _ => std::panic!("unsupported {:?}", inst.ops[0]),
                    };
                    wave.print();
                    valu.busy = false;
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

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test1() {
        let config = GPUConfig {
            DRAM_latency: 300,
            DRAM_bandwidth: 256,
            L1_size: 1 << 14,
            L1_latency: 100,
            L2_size: 1 << 15,
            L2_latency: 200,
            sampler_cache_size: 1 << 10,
            sampler_latency: 100,
            samplers_per_cu: 1,
            sampler_cache_latency: 100,
            SLM_size: 1 << 14,
            SLM_latency: 20,
            SLM_banks: 32,
            VGPRF_per_pe: 16,
            SGPRF_per_wave: 16,
            wave_size: 1,
            CU_count: 4,
            ALU_per_cu: 4,
            waves_per_cu: 16,
            fd_per_cu: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let mut program = Program { ins: Vec::new() };
        // 1: mov r1.xyzw, vec4(1.0f, 1.0f, 1.0f, 0.0f)
        // 2: mov r2.xyzw, vec4(1.0f, 2.0f, 3.0f, 4.0f)
        // 3: add r0.xyzw, r1.xyzw, r2.xyzw
        program.ins.push(Instruction {
            ty: InstTy::MOV,
            line: 1,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::Immediate(ImmediateVal::V4F(1.0, 1.0, 1.0, 0.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            ty: InstTy::MOV,
            line: 2,
            ops: [
                Operand::VRegister(RegRef {
                    id: 2,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::Immediate(ImmediateVal::V4F(1.0, 2.0, 3.0, 4.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            ty: InstTy::ADD,
            line: 3,
            ops: [
                Operand::VRegister(RegRef {
                    id: 0,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::VRegister(RegRef {
                    id: 2,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::NONE,
            ],
        });
        // @Test
        dispatch(&mut gpu_state, program, 32, 1);
        clock(&mut gpu_state);
        clock(&mut gpu_state);
        clock(&mut gpu_state);
    }
    #[test]
    fn test2() {}
}
