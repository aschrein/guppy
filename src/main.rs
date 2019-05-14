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

fn applyReadSwizzle(val: &Value, regref: &RegRef) -> Value {
    let mut out: Value = [0, 0, 0, 0];
    for i in 0..4 {
        match regref.comps[i] {
            Component::X => out[i] = val[0],
            Component::Y => out[i] = val[1],
            Component::Z => out[i] = val[2],
            Component::W => out[i] = val[3],
            Component::NONE => {}
        }
    }
    out
}
fn applyWriteSwizzle(out: &mut Value, val: &Value, regref: &RegRef) {
    let mut out_: Value = *out;
    for i in 0..4 {
        match regref.comps[i] {
            Component::X => out_[0] = val[i],
            Component::Y => out_[1] = val[i],
            Component::Z => out_[2] = val[i],
            Component::W => out_[3] = val[i],
            Component::NONE => {}
        }
    }
    *out = out_;
}

fn AddU32(this: &Value, that: &Value) -> Value {
    [
        this[0] + that[0],
        this[1] + that[1],
        this[2] + that[2],
        this[3] + that[3],
    ]
}

fn SubU32(this: &Value, that: &Value) -> Value {
    [
        this[0] - that[0],
        this[1] - that[1],
        this[2] - that[2],
        this[3] - that[3],
    ]
}

fn MulU32(this: &Value, that: &Value) -> Value {
    [
        this[0] * that[0],
        this[1] * that[1],
        this[2] * that[2],
        this[3] * that[3],
    ]
}

fn DivU32(this: &Value, that: &Value) -> Value {
    [
        this[0] / that[0],
        this[1] / that[1],
        this[2] / that[2],
        this[3] / that[3],
    ]
}

fn LTU32(this: &Value, that: &Value) -> Value {
    [
        if this[0] < that[0] { 1 } else { 0 },
        if this[1] < that[1] { 1 } else { 0 },
        if this[2] < that[2] { 1 } else { 0 },
        if this[3] < that[3] { 1 } else { 0 },
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

fn SubF32(this_: &Value, that_: &Value) -> Value {
    let this = castToFValue(this_);
    let that = castToFValue(that_);
    castToValue(&[
        this[0] - that[0],
        this[1] - that[1],
        this[2] - that[2],
        this[3] - that[3],
    ])
}

fn MulF32(this_: &Value, that_: &Value) -> Value {
    let this = castToFValue(this_);
    let that = castToFValue(that_);
    castToValue(&[
        this[0] * that[0],
        this[1] * that[1],
        this[2] * that[2],
        this[3] * that[3],
    ])
}

fn DivF32(this_: &Value, that_: &Value) -> Value {
    let this = castToFValue(this_);
    let that = castToFValue(that_);
    castToValue(&[
        this[0] / that[0],
        this[1] / that[1],
        this[2] / that[2],
        this[3] / that[3],
    ])
}

fn LTF32(this_: &Value, that_: &Value) -> Value {
    let this = castToFValue(this_);
    let that = castToFValue(that_);
    [
        if this[0] < that[0] { 1 } else { 0 },
        if this[1] < that[1] { 1 } else { 0 },
        if this[2] < that[2] { 1 } else { 0 },
        if this[3] < that[3] { 1 } else { 0 },
    ]
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

fn U2F(val: &Value) -> Value {
    castToValue(&[val[0] as f32, val[1] as f32, val[2] as f32, val[3] as f32])
}

use std::rc::Rc;

type ExecMask = Vec<bool>;
type VGPRF = Vec<Register>;
type SGPRF = Vec<Register>;

struct WaveState {
    // Array of vector registers, vgprfs[0] means array of r0 for each laneS
    vgprfs: Vec<VGPRF>,
    sgprf: SGPRF,
    // id of the group within the same dispatch request
    group_id: u32,
    group_size: u32,
    // id within the same dispatch group
    wave_id: u32,
    // Shared pointer to the program
    program: Option<Rc<Program>>,
    // Next instruction
    pc: u32,
    // Determines temporarly disabled lanes
    exec_mask: ExecMask,
    // Determines dead lanes
    live_mask: ExecMask,
    // execution mask stack
    exec_mask_stack: Vec<(ExecMask, u32)>,

    // if this wave has been stalled on the previous cycle
    stalled: bool,
    // if this wave executes some job
    enabled: bool,
    // if this wave is being dispatched on this cycle
    has_been_dispatched: bool,
}

impl WaveState {
    fn print(&self) {
        println!(
            "WaveState wave_id::{}, group_id::{}",
            self.wave_id, self.group_id
        );
        println!("exec_mask::{:?}", self.exec_mask);

        for vreg in &self.vgprfs {
            println!(
                "{:?}",
                vreg.iter()
                    .map(|x| castToFValue(&x.val))
                    .collect::<Vec<FValue>>()
            );
        }
    }
    fn new(config: &GPUConfig) -> WaveState {
        WaveState {
            vgprfs: Vec::new(),
            sgprf: Vec::new(),
            wave_id: 0,
            group_id: 0,
            group_size: 0,
            program: None,
            pc: 0,
            exec_mask: Vec::new(),
            live_mask: Vec::new(),
            exec_mask_stack: Vec::new(),
            stalled: false,
            enabled: false,
            has_been_dispatched: false,
        }
    }
    fn getValues(&self, op: &Operand) -> Vec<Value> {
        let wave_width = self.vgprfs[0].len();
        match &op {
            Operand::VRegister(vop) => self.vgprfs[vop.id as usize]
                .iter()
                .map(|r| applyReadSwizzle(&r.val, &vop))
                .collect::<Vec<Value>>(),
            Operand::Immediate(imm) => match imm {
                ImmediateVal::V4F(x, y, z, w) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push(castToValue(&[*x, *y, *z, *w]));
                    }
                    values
                }
                ImmediateVal::V3F(x, y, z) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push(castToValue(&[*x, *y, *z, 0.0]));
                    }
                    values
                }
                ImmediateVal::V2F(x, y) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push(castToValue(&[*x, *y, 0.0, 0.0]));
                    }
                    values
                }
                ImmediateVal::V1F(x) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push(castToValue(&[*x, 0.0, 0.0, 0.0]));
                    }
                    values
                }
                ImmediateVal::V4U(x, y, z, w) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push([*x, *y, *z, *w]);
                    }
                    values
                }
                ImmediateVal::V3U(x, y, z) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push([*x, *y, *z, 0]);
                    }
                    values
                }
                ImmediateVal::V2U(x, y) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push([*x, *y, 0, 0]);
                    }
                    values
                }
                ImmediateVal::V1U(x) => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        values.push([*x, 0, 0, 0]);
                    }
                    values
                }
                _ => std::panic!(""),
            },
            Operand::Builtin(imm) => match imm {
                BuiltinVal::THREAD_ID => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        let id = self.group_id * self.group_size
                            + self.wave_id * wave_width as u32
                            + i as u32;
                        values.push([id, id, id, id]);
                    }
                    values
                }
                BuiltinVal::LANE_ID => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        let id = i as u32;
                        values.push([id, id, id, id]);
                    }
                    values
                }
                BuiltinVal::WAVE_ID => {
                    let mut values: Vec<Value> = Vec::new();
                    for i in 0..wave_width {
                        let id = self.wave_id;
                        values.push([id, id, id, id]);
                    }
                    values
                }

                _ => std::panic!(""),
            },
            // @TODO: immediate
            _ => std::panic!(""),
        }
    }
    fn dispatch(
        &mut self,
        config: &GPUConfig,
        program: &Rc<Program>,
        wave_id: u32,
        group_id: u32,
        group_size: u32,
    ) {
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
            exec_mask.push((i + wave_id * config.wave_size) < group_size);
        }
        self.program = Some(program.clone());
        self.sgprf = sgprf;
        self.vgprfs = vgprfs;
        self.exec_mask = exec_mask;
        self.stalled = false;
        self.pc = 0;
        self.wave_id = wave_id;
        self.group_id = group_id;
        self.group_size = group_size;
        self.has_been_dispatched = false;
        self.enabled = true;
    }
}

struct VALUState {
    instr: Option<Instruction>,
    exec_mask: Option<Vec<bool>>,
    wave_id: u32,
    timer: u32,
    busy: bool,
    was_busy: bool,
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
                was_busy: false,
                exec_mask: None,
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
    clock_counter: u32,
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
            clock_counter: 0,
        }
    }
    fn findFreeWave(&self) -> Option<(usize, usize)> {
        for (i, cu) in self.cus.iter().enumerate() {
            for (j, wave) in cu.waves.iter().enumerate() {
                if !wave.enabled {
                    return Some((i, j));
                }
            }
        }
        None
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
enum Interpretation {
    F32,
    I32,
    U32,
    NONE,
}

// r1.__xy
#[derive(Debug, Clone, PartialEq)]
struct RegRef {
    id: u32,
    comps: [Component; 4],
    //interp: Interpretation,
}

#[derive(Debug, Clone, PartialEq)]
struct BufferRef {
    id: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct TextureRef {
    id: u32,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
enum BuiltinVal {
    // Global thread id
    THREAD_ID,
    WAVE_ID,
    // ID within wave
    LANE_ID,
}

#[derive(Debug, Clone, PartialEq)]
enum Operand {
    VRegister(RegRef),
    SRegister(RegRef),
    Buffer(BufferRef),
    Texture(TextureRef),
    Immediate(ImmediateVal),
    Builtin(BuiltinVal),
    Label(u32),
    NONE,
}

#[derive(Debug, Clone, PartialEq)]
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
    UTOF,
    LT,
    BR_PUSH,
    PUSH_MASK,
    POP_MASK,
    MASK_NZ,
    JMP,
    RET,
    NONE,
}

#[derive(Debug, Clone, PartialEq)]
struct Instruction {
    ty: InstTy,
    interp: Interpretation,
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

#[derive(Clone, Debug)]
struct Program {
    ins: Vec<Instruction>,
}

struct DispatchReq {
    program: Rc<Program>,
    group_size: u32,
    group_count: u32,
}

fn dispatch(gpu_state: &mut GPUState, program: &Program, group_size: u32, group_count: u32) {
    let disp_req = DispatchReq {
        program: Rc::new(program.clone()),
        group_count: group_count,
        group_size: group_size,
    };
    gpu_state.dreqs.push(disp_req);
}

fn clock(gpu_state: &mut GPUState) -> bool {
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
                    // @TODO: dispatch groups separately
                    // Make sure it's a power of two and no greater than 1024
                    assert!(
                        ((req.group_size - 1) & req.group_size) == 0
                            && req.group_size <= 1024
                            && req.group_count * req.group_size != 0
                    );

                    // @DEAD
                    // assert!(req.group_count * req.group_size % gpu_state.config.wave_size == 0);

                    let warp_count = req.group_count * req.group_size / gpu_state.config.wave_size;
                    if warp_count > cnt_free_waves {
                        deferredReq.push(req);
                        continue;
                    }
                    cnt_free_waves -= warp_count;
                    // Find a free wave and dispatch on it
                    for group_id in 0..req.group_count {
                        for wave_id in 0..((req.group_size + gpu_state.config.wave_size - 1)
                            / gpu_state.config.wave_size)
                        {
                            let wave_path = gpu_state.findFreeWave().unwrap();
                            let config = &gpu_state.config;
                            gpu_state.cus[wave_path.0].waves[wave_path.1].dispatch(
                                config,
                                &req.program,
                                wave_id,
                                group_id,
                                req.group_size,
                            );
                        }
                    }
                }
                None => break,
            }
        }
        gpu_state.dreqs.append(&mut deferredReq);
    }
    let mut didSomeWork = false;
    // @Fetch-Submit part
    // @TODO: Refactor the loop - it looks ugly
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
                        // At least we have some wave that must be doing something
                        didSomeWork = true;
                        // This wave might be stalled because of instruction dependencies
                        // or it might be waiting for the last instruction before retiring

                        // @DEAD
                        // if wave.pc >= wave.program.as_ref().unwrap().ins.len() as u32 {
                        //     let has_locks = wave.vgprfs.iter().any(|x| x.iter().any(|y| y.locked));
                        //     if has_locks {
                        //         wave.stalled = true;
                        //         continue;
                        //     }
                        //     // Retire if no register locks
                        //     wave.enabled = false;
                        // }

                        let inst = &(*wave.program.as_ref().unwrap()).ins[wave.pc as usize];
                        let mut hasVRegOps = false;
                        let mut hasSRegOps = false;
                        let mut dispatchOnSampler = false;
                        let mut control_flow_cmd = false;

                        let mut has_dep = false;
                        // Do simple decoding and sanity checks
                        for op in &inst.ops {
                            match &op {
                                Operand::VRegister(vreg) => {
                                    for (i, item) in
                                        wave.vgprfs[vreg.id as usize].iter_mut().enumerate()
                                    {
                                        if wave.exec_mask[i] && item.locked {
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
                                Operand::Builtin(buiiltin) => {}
                                Operand::Label(label) => {}
                                Operand::NONE => {}
                                _ => {
                                    std::panic!("");
                                }
                            }
                        }
                        match &inst.ty {
                            InstTy::ADD | InstTy::SUB | InstTy::MUL | InstTy::DIV | InstTy::LT => {
                                inst.assertThreeOp();
                            }
                            InstTy::BR_PUSH
                            | InstTy::POP_MASK
                            | InstTy::RET
                            | InstTy::PUSH_MASK
                            | InstTy::JMP
                            | InstTy::MASK_NZ => {
                                control_flow_cmd = true;
                            }
                            InstTy::MOV | InstTy::UTOF => {
                                inst.assertTwoOp();
                            }
                            _ => {
                                std::panic!("");
                            }
                        };
                        // @TODO: Make proper sanity checks

                        // @DEAD
                        // assert!(
                        //     vec![hasSRegOps, dispatchOnSampler, hasVRegOps]
                        //         .iter()
                        //         .filter(|&a| *a == true)
                        //         .collect::<Vec<&bool>>()
                        //         .len()
                        //         == 1 // Mixing different register types
                        //         || branchingOp
                        // );

                        // One of the operands is being locked
                        // Stall the wave
                        if has_dep {
                            wave.stalled = true;
                            continue;
                        }
                        let has_locks = wave.vgprfs.iter().any(|x| x.iter().any(|y| y.locked));
                        if let InstTy::RET = inst.ty {
                            // If the wave has locks then stall on the return statement
                            // Don't increment the pc - we will hit the same instruction next cycle
                            if has_locks {
                                wave.stalled = true;
                                continue;
                            }
                            wave.enabled = false;
                            continue;
                        } else if let InstTy::POP_MASK = inst.ty {
                            assert!(wave.exec_mask_stack.len() != 0);
                            let prev_mask = wave.exec_mask_stack.pop().unwrap();
                            wave.exec_mask = prev_mask.0;
                            wave.pc = prev_mask.1;
                            wave.has_been_dispatched = true;
                        } else if let InstTy::PUSH_MASK = inst.ty {
                            let converge_addr = match &inst.ops[0] {
                                Operand::Label(ca) => *ca,
                                _ => std::panic!(""),
                            };
                            wave.exec_mask_stack
                                .push((wave.exec_mask.clone(), converge_addr));
                            wave.pc += 1;
                            wave.has_been_dispatched = true;
                        } else if let InstTy::JMP = inst.ty {
                            let converge_addr = match &inst.ops[0] {
                                Operand::Label(ca) => *ca,
                                _ => std::panic!(""),
                            };
                            wave.pc = converge_addr;
                            wave.has_been_dispatched = true;
                        } else if let InstTy::BR_PUSH = inst.ty {
                            match &inst.ops[0] {
                                Operand::VRegister(vreg) => {
                                    match &vreg.comps {
                                        [x, Component::NONE, Component::NONE, Component::NONE] => {

                                        }
                                        _ => std::panic!("The first parameter should be one component e.g. r1.x, not r1.xy")
                                    }
                                    let values = wave.getValues(&inst.ops[0]);
                                    let true_mask = values
                                        .iter()
                                        .map(|i| i[0] != 0)
                                        .zip(&wave.exec_mask)
                                        .map(|(a, b): (bool, &bool)| a && *b)
                                        .collect::<Vec<_>>();
                                    let false_mask = true_mask
                                        .iter()
                                        .map(|i| !i)
                                        .zip(&wave.exec_mask)
                                        .map(|(a, b): (bool, &bool)| a && *b)
                                        .collect::<Vec<_>>();
                                    let (false_addr, converge_addr) =
                                        match (&inst.ops[1], &inst.ops[2]) {
                                            (Operand::Label(fa), Operand::Label(ca)) => (*fa, *ca),
                                            _ => std::panic!(""),
                                        };
                                    wave.exec_mask_stack
                                        .push((wave.exec_mask.clone(), converge_addr));
                                    wave.exec_mask_stack.push((false_mask, false_addr));
                                    wave.exec_mask = true_mask;
                                    wave.pc += 1;
                                    wave.has_been_dispatched = true;
                                }
                                _ => std::panic!("Unsupported branch parameter"),
                            }
                        } else if let InstTy::MASK_NZ = inst.ty {
                            match &inst.ops[0] {
                                Operand::VRegister(vreg) => {
                                    match &vreg.comps {
                                        [x, Component::NONE, Component::NONE, Component::NONE] => {

                                        }
                                        _ => std::panic!("The first parameter should be one component e.g. r1.x, not r1.xy")
                                    }
                                    let values = wave.getValues(&inst.ops[0]);
                                    let true_mask = values
                                        .iter()
                                        .map(|i| i[0] != 0)
                                        .zip(&wave.exec_mask)
                                        .map(|(a, b): (bool, &bool)| a && *b)
                                        .collect::<Vec<_>>();
                                    let nz_cnt =
                                        true_mask.iter().filter(|&x| *x).collect::<Vec<_>>().len();
                                    if nz_cnt == 0 {
                                        assert!(wave.exec_mask_stack.len() != 0);
                                        let prev_mask = wave.exec_mask_stack.pop().unwrap();
                                        wave.exec_mask = prev_mask.0;
                                        wave.pc = prev_mask.1;
                                    } else {
                                        wave.exec_mask = true_mask;
                                        wave.pc += 1;
                                    }
                                    wave.has_been_dispatched = true;
                                }
                                _ => std::panic!("Unsupported parameter"),
                            }
                        }
                        // For simplicity dispatch commands basing only on registers used
                        else if hasVRegOps {
                            for valu in &mut cu.valus {
                                if valu.busy {
                                    continue;
                                }
                                wave.has_been_dispatched = true;
                                valu.instr = Some(inst.clone());
                                // @TODO: determine the timer value
                                valu.timer = getLatency(&inst.ty);
                                valu.busy = true;
                                valu.wave_id = wave_id as u32;
                                valu.exec_mask = Some(wave.exec_mask.clone());
                                // Lock the destination registers
                                match &inst.ty {
                                    InstTy::ADD
                                    | InstTy::SUB
                                    | InstTy::MUL
                                    | InstTy::DIV
                                    | InstTy::LT => {
                                        if let Operand::VRegister(dst) = &inst.ops[0] {
                                            for (i, item) in
                                                wave.vgprfs[dst.id as usize].iter_mut().enumerate()
                                            {
                                                if wave.exec_mask[i] {
                                                    item.locked = true;
                                                }
                                            }
                                        } else {
                                            std::panic!("")
                                        }
                                    }
                                    InstTy::MOV | InstTy::UTOF => {
                                        if let Operand::VRegister(dst) = &inst.ops[0] {
                                            for (i, item) in
                                                wave.vgprfs[dst.id as usize].iter_mut().enumerate()
                                            {
                                                if wave.exec_mask[i] {
                                                    item.locked = true;
                                                }
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
                            // If an instruction was issued then increment PC
                            if wave.has_been_dispatched {
                                wave.pc += 1;
                                if wave.pc as usize == wave.program.as_ref().unwrap().ins.len() {
                                    // If the wave has locks then stall on the return statement
                                    if has_locks {
                                        wave.stalled = true;
                                        continue;
                                    }

                                    wave.enabled = false;
                                    // Wave has been retired
                                }
                            } else {
                                // Not enough resources to dispatch a command
                            }
                        } else if hasSRegOps {
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
                valu.was_busy = false;
                if !valu.busy {
                    continue;
                }
                didSomeWork = true;
                valu.was_busy = true;
                if valu.timer != 0 {
                    valu.timer -= 1;
                }
                if valu.timer == 0 {
                    let inst = &valu.instr.as_ref().unwrap();
                    let mut wave = &mut cu.waves[valu.wave_id as usize];
                    let exec_mask = &valu.exec_mask.as_ref().unwrap();
                    match &inst.ty {
                        InstTy::ADD | InstTy::SUB | InstTy::MUL | InstTy::DIV | InstTy::LT => {
                            let src1 = wave.getValues(&inst.ops[1]);
                            let src2 = wave.getValues(&inst.ops[2]);
                            let dst = match &inst.ops[0] {
                                Operand::VRegister(dst) => dst,
                                _ => std::panic!(""),
                            };
                            let result = {
                                // @TODO: sub, div, mul
                                src1.iter()
                                    .zip(src2.iter())
                                    // @TODO: Support different types
                                    .map(|(&x1, &x2)| match inst.interp {
                                        Interpretation::F32 => match inst.ty {
                                            InstTy::ADD => AddF32(&x1, &x2),
                                            InstTy::SUB => SubF32(&x1, &x2),
                                            InstTy::MUL => MulF32(&x1, &x2),
                                            InstTy::DIV => DivF32(&x1, &x2),
                                            InstTy::LT => LTF32(&x1, &x2),
                                            _ => std::panic!(""),
                                        },
                                        Interpretation::U32 => match inst.ty {
                                            InstTy::ADD => AddU32(&x1, &x2),
                                            InstTy::SUB => SubU32(&x1, &x2),
                                            InstTy::MUL => MulU32(&x1, &x2),
                                            InstTy::DIV => DivU32(&x1, &x2),
                                            InstTy::LT => LTU32(&x1, &x2),
                                            _ => std::panic!(""),
                                        },

                                        _ => std::panic!(""),
                                    })
                                    .collect::<Vec<Value>>()
                            };
                            assert!(wave.vgprfs[dst.id as usize].len() == result.len());
                            // Registers should be unlocked by this time
                            for (i, item) in
                                &mut wave.vgprfs[dst.id as usize].iter_mut().enumerate()
                            {
                                if exec_mask[i as usize] {
                                    applyWriteSwizzle(&mut item.val, &result[i], &dst);
                                    item.locked = false;
                                }
                            }
                        }
                        InstTy::MOV | InstTy::UTOF => {
                            let src1 = wave.getValues(&inst.ops[1]);
                            let dst = match &inst.ops[0] {
                                Operand::VRegister(dst) => dst,
                                _ => std::panic!(""),
                            };
                            assert!(wave.vgprfs[dst.id as usize].len() == src1.len());
                            // Registers should be unlocked by this time
                            for (i, item) in
                                &mut wave.vgprfs[dst.id as usize].iter_mut().enumerate()
                            {
                                if exec_mask[i as usize] {
                                    let src = match &inst.ty {
                                        InstTy::MOV => src1[i],
                                        InstTy::UTOF => U2F(&src1[i]),
                                        _ => std::panic!(""),
                                    };
                                    applyWriteSwizzle(&mut item.val, &src, &dst);
                                    item.locked = false;
                                }
                            }
                        }
                        _ => std::panic!("unsupported {:?}", inst.ops[0]),
                    };
                    // wave.print();
                    valu.busy = false;
                }
            }
            // And on Scalar ALUs
            for salu in &mut cu.salus {}
        }
    }
    gpu_state.clock_counter += 1;
    didSomeWork
}

#[macro_use]
extern crate lazy_static;
extern crate regex;
use regex::Regex;

fn parse(text: &str) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::new();
    lazy_static! {
        static ref VRegRE: Regex = Regex::new(r"r([0-9]+)\.([xyzw]+)").unwrap();
        static ref V4FRE: Regex =
            Regex::new(r"vec4[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V2FRE: Regex = Regex::new(r"vec2[ ]*\([ ]*([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V3FRE: Regex =
            Regex::new(r"vec3[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V1FRE: Regex = Regex::new(r"vec1[ ]*\([ ]*([^ ]+)[ ]*\)").unwrap();
        static ref V4URE: Regex =
            Regex::new(r"uvec4[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V2URE: Regex = Regex::new(r"uvec2[ ]*\([ ]*([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V3URE: Regex =
            Regex::new(r"uvec3[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V1URE: Regex = Regex::new(r"uvec1[ ]*\([ ]*([^ ]+)[ ]*\)").unwrap();
        static ref spaceRE: Regex = Regex::new(r"[ ]+").unwrap();
        static ref garbageRE: Regex = Regex::new(r"^[ ]+|[ ]+$|[\t]+|;.*").unwrap();
        static ref labelRE: Regex = Regex::new(r"^[ ]*([^ ]+)[ ]*:[ ]*").unwrap();
    }
    let mapSwizzle = |c: char| -> Component {
        match c {
            'x' => Component::X,
            'y' => Component::Y,
            'z' => Component::Z,
            'w' => Component::W,
            _ => std::panic!(""),
        }
    };
    let parseOperand = |s: &str| -> Operand {
        if let Some(x) = VRegRE.captures(s) {
            return Operand::VRegister({
                let regnum = x.get(1).unwrap().as_str();
                let mut swizzle = x
                    .get(2)
                    .unwrap()
                    .as_str()
                    .chars()
                    .map(|c| mapSwizzle(c))
                    .collect::<Vec<_>>();
                while swizzle.len() < 4 {
                    swizzle.push(Component::NONE);
                }
                RegRef {
                    id: regnum.parse::<u32>().unwrap(),
                    comps: [
                        swizzle[0].clone(),
                        swizzle[1].clone(),
                        swizzle[2].clone(),
                        swizzle[3].clone(),
                    ],
                }
            });
        } else if let Some(x) = V4URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V4U(
                    x.get(1).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(3).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(4).unwrap().as_str().parse::<u32>().unwrap(),
                )
            });
        } else if let Some(x) = V3URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V3U(
                    x.get(1).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(4).unwrap().as_str().parse::<u32>().unwrap(),
                )
            });
        } else if let Some(x) = V2URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V2U(
                    x.get(1).unwrap().as_str().parse::<u32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<u32>().unwrap(),
                )
            });
        } else if let Some(x) = V1URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V1U(x.get(1).unwrap().as_str().parse::<u32>().unwrap())
            });
        } else if let Some(x) = V4FRE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V4F(
                    x.get(1).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(3).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(4).unwrap().as_str().parse::<f32>().unwrap(),
                )
            });
        } else if let Some(x) = V3FRE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V3F(
                    x.get(1).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(4).unwrap().as_str().parse::<f32>().unwrap(),
                )
            });
        } else if let Some(x) = V2FRE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V2F(
                    x.get(1).unwrap().as_str().parse::<f32>().unwrap(),
                    x.get(2).unwrap().as_str().parse::<f32>().unwrap(),
                )
            });
        } else if let Some(x) = V1FRE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V1F(x.get(1).unwrap().as_str().parse::<f32>().unwrap())
            });
        } else if s == "thread_id" {
            return Operand::Builtin(BuiltinVal::THREAD_ID);
        } else if s == "lane_id" {
            return Operand::Builtin(BuiltinVal::LANE_ID);
        } else if s == "wave_id" {
            return Operand::Builtin(BuiltinVal::WAVE_ID);
        }

        std::panic!(format!("unrecognized {:?}", s))
    };
    let lines = text.split("\n").enumerate().collect::<Vec<(usize, &str)>>();
    let mut instructions: Vec<(usize, String)> = Vec::new();
    use std::collections::HashMap;
    let mut label_map: HashMap<String, usize> = HashMap::new();
    for (line_num, line) in lines {
        let normalized = garbageRE.replace_all(line.clone(), " ");
        if let Some(s) = labelRE.captures(&normalized) {
            // Label points to the next line
            label_map.insert(String::from(s.get(1).unwrap().as_str()), line_num + 1);
        } else {
            instructions.push((line_num, String::from(normalized)));
        }
    }
    for (line_num, line) in instructions {
        // take first after split(" ")
        let parts = spaceRE
            .replace_all(&line, " ")
            .split(" ")
            .map(|s| String::from(s))
            .filter(|s| s != "")
            .next();
        //.collect::<Vec<String>>();
        if parts.is_none() {
            continue;
        }
        let command = parts.unwrap();
        let operands = &line[command.len() + 1..]
            .split(",")
            .map(|s| String::from(garbageRE.replace_all(s, "")))
            .filter(|s| s != "")
            .collect::<Vec<String>>();
        // println!("{:?}", operands);
        let instr = match command.as_str() {
            "mov" | "utof" => {
                assert!(operands.len() == 2);
                let dstRef = parseOperand(&operands[0]);
                let srcRef = parseOperand(&operands[1]);
                Instruction {
                    ty: match command.as_str() {
                        "mov" => InstTy::MOV,
                        "utof" => InstTy::UTOF,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [dstRef, srcRef, Operand::NONE, Operand::NONE],
                }
            }
            "mask_nz" => {
                assert!(operands.len() == 1);
                let dstRef = parseOperand(&operands[0]);
                Instruction {
                    ty: match command.as_str() {
                        "mask_nz" => InstTy::MASK_NZ,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [dstRef, Operand::NONE, Operand::NONE, Operand::NONE],
                }
            }
            "pop_mask" | "ret" => {
                assert!(operands.len() == 0);
                Instruction {
                    ty: match command.as_str() {
                        "pop_mask" => InstTy::POP_MASK,
                        "ret" => InstTy::RET,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE],
                }
            }

            "add_f32" | "sub_f32" | "mul_f32" | "div_f32" | "lt_f32" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let src1Ref = parseOperand(&operands[1]);
                let src2Ref = parseOperand(&operands[2]);
                Instruction {
                    ty: match command.as_str() {
                        "add_f32" => InstTy::ADD,
                        "sub_f32" => InstTy::SUB,
                        "mul_f32" => InstTy::MUL,
                        "div_f32" => InstTy::DIV,
                        "lt_f32" => InstTy::LT,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::F32,
                    line: line_num as u32,
                    ops: [dstRef, src1Ref, src2Ref, Operand::NONE],
                }
            }
            "add_u32" | "sub_u32" | "mul_u32" | "div_u32" | "lt_u32" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let src1Ref = parseOperand(&operands[1]);
                let src2Ref = parseOperand(&operands[2]);
                Instruction {
                    ty: match command.as_str() {
                        "add_u32" => InstTy::ADD,
                        "sub_u32" => InstTy::SUB,
                        "mul_u32" => InstTy::MUL,
                        "div_u32" => InstTy::DIV,
                        "lt_u32" => InstTy::LT,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::U32,
                    line: line_num as u32,
                    ops: [dstRef, src1Ref, src2Ref, Operand::NONE],
                }
            }

            "jmp" | "push_mask" => {
                assert!(operands.len() == 1);
                let label = label_map.get(&operands[0]).unwrap();
                Instruction {
                    ty: match command.as_str() {
                        "jmp" => InstTy::JMP,
                        "push_mask" => InstTy::PUSH_MASK,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [
                        Operand::Label(*label as u32),
                        Operand::NONE,
                        Operand::NONE,
                        Operand::NONE,
                    ],
                }
            }

            "br_push" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let else_label = label_map.get(&operands[1]).unwrap();
                let converge_label = label_map.get(&operands[2]).unwrap();
                Instruction {
                    ty: InstTy::BR_PUSH,
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [
                        dstRef,
                        Operand::Label(*else_label as u32),
                        Operand::Label(*converge_label as u32),
                        Operand::NONE,
                    ],
                }
            }

            _ => std::panic!("unrecognized command"),
        };
        out.push(instr);
    }
    // Resolve labels to instruction indices
    let mut line_map: HashMap<u32, u32> = HashMap::new();
    for (i, inst) in out.iter().enumerate() {
        line_map.insert(inst.line, i as u32);
    }
    for inst in &mut out {
        if InstTy::BR_PUSH == inst.ty {
            let new_indices = match (&inst.ops[1], &inst.ops[2]) {
                (Operand::Label(else_label), Operand::Label(converge_label)) => (
                    *line_map.get(else_label).unwrap(),
                    *line_map.get(converge_label).unwrap(),
                ),
                _ => panic!(""),
            };
            inst.ops[1] = Operand::Label(new_indices.0);
            inst.ops[2] = Operand::Label(new_indices.1);
        }
        if InstTy::JMP == inst.ty || InstTy::PUSH_MASK == inst.ty {
            let new_indices = match &inst.ops[0] {
                Operand::Label(label) => (*line_map.get(label).unwrap(),),
                _ => panic!(""),
            };
            inst.ops[0] = Operand::Label(new_indices.0);
        }
    }
    out
}

fn main() {
    println!("Hello, world!");
}

// @TODO: Probably should be a part of gpu_config
// also integer ops should be less expensive than the corresponding floating point ones
fn getLatency(ty: &InstTy) -> u32 {
    match &ty {
        InstTy::ADD | InstTy::SUB | InstTy::MOV | InstTy::UTOF | InstTy::LT => 1,
        InstTy::MUL => 2,
        InstTy::DIV => 4,
        _ => panic!(""),
    }
}

enum Event {
    GROUP_DISPATCHED((u32, Vec<(u32, u32)>)),
    WAVE_STALLED((u32, u32)),
    WAVE_RETIRED((u32, u32)),
    GROUP_RETIRED(u32),
    INST_DISPATCHED((u32, u32)),
    INST_RETIRED((u32, u32)),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_mask_test_2() {
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
            VGPRF_per_pe: 8,
            SGPRF_per_wave: 16,
            wave_size: 32,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
                mov r0.x, lane_id
                mov r1.x, uvec1(0)
                push_mask LOOP_END
            LOOP_PROLOG:
                lt_u32 r0.y, r0.x, uvec1(16)
                add_u32 r0.x, r0.x, uvec1(1)
                mask_nz r0.y
            LOOP_BEGIN:
                add_u32 r1.x, r1.x, uvec1(1)
                jmp LOOP_PROLOG
            LOOP_END:
                ret
                ",
            ),
        };
        dispatch(&mut gpu_state, &program, 16, 1);
        let mut mask_history: Vec<String> = Vec::new();
        while clock(&mut gpu_state) {
            for cu in &gpu_state.cus {
                for wave in &cu.waves {
                    if wave.enabled {
                        mask_history.push(
                            wave.exec_mask
                                .iter()
                                .map(|&b| if b { '1' } else { '0' })
                                .collect(),
                        );
                        // print!("\"");
                        // for bit in &wave.exec_mask {
                        //     print!("{}", if *bit { 1 } else { 0 });
                        // }
                        // print!("\",");
                    }
                }
            }
            // println!("");
        }
        // println!(
        //     "{:?}",
        //     gpu_state.cus[0].waves[0].vgprfs[1]
        //         .iter()
        //         .map(|r| &r.val[0])
        //         .collect::<Vec<_>>()
        // );
        assert_eq!(
            vec![
                16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0
            ],
            gpu_state.cus[0].waves[0].vgprfs[1]
                .iter()
                .map(|r| r.val[0])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mask_history,
            vec![
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111110000000000000000",
                "11111111111111100000000000000000",
                "11111111111111100000000000000000",
                "11111111111111100000000000000000",
                "11111111111111100000000000000000",
                "11111111111111100000000000000000",
                "11111111111111000000000000000000",
                "11111111111111000000000000000000",
                "11111111111111000000000000000000",
                "11111111111111000000000000000000",
                "11111111111111000000000000000000",
                "11111111111110000000000000000000",
                "11111111111110000000000000000000",
                "11111111111110000000000000000000",
                "11111111111110000000000000000000",
                "11111111111110000000000000000000",
                "11111111111100000000000000000000",
                "11111111111100000000000000000000",
                "11111111111100000000000000000000",
                "11111111111100000000000000000000",
                "11111111111100000000000000000000",
                "11111111111000000000000000000000",
                "11111111111000000000000000000000",
                "11111111111000000000000000000000",
                "11111111111000000000000000000000",
                "11111111111000000000000000000000",
                "11111111110000000000000000000000",
                "11111111110000000000000000000000",
                "11111111110000000000000000000000",
                "11111111110000000000000000000000",
                "11111111110000000000000000000000",
                "11111111100000000000000000000000",
                "11111111100000000000000000000000",
                "11111111100000000000000000000000",
                "11111111100000000000000000000000",
                "11111111100000000000000000000000",
                "11111111000000000000000000000000",
                "11111111000000000000000000000000",
                "11111111000000000000000000000000",
                "11111111000000000000000000000000",
                "11111111000000000000000000000000",
                "11111110000000000000000000000000",
                "11111110000000000000000000000000",
                "11111110000000000000000000000000",
                "11111110000000000000000000000000",
                "11111110000000000000000000000000",
                "11111100000000000000000000000000",
                "11111100000000000000000000000000",
                "11111100000000000000000000000000",
                "11111100000000000000000000000000",
                "11111100000000000000000000000000",
                "11111000000000000000000000000000",
                "11111000000000000000000000000000",
                "11111000000000000000000000000000",
                "11111000000000000000000000000000",
                "11111000000000000000000000000000",
                "11110000000000000000000000000000",
                "11110000000000000000000000000000",
                "11110000000000000000000000000000",
                "11110000000000000000000000000000",
                "11110000000000000000000000000000",
                "11100000000000000000000000000000",
                "11100000000000000000000000000000",
                "11100000000000000000000000000000",
                "11100000000000000000000000000000",
                "11100000000000000000000000000000",
                "11000000000000000000000000000000",
                "11000000000000000000000000000000",
                "11000000000000000000000000000000",
                "11000000000000000000000000000000",
                "11000000000000000000000000000000",
                "10000000000000000000000000000000",
                "10000000000000000000000000000000",
                "10000000000000000000000000000000",
                "10000000000000000000000000000000",
                "10000000000000000000000000000000",
                "11111111111111110000000000000000",
            ]
        )
    }

    #[test]
    fn exec_mask_test() {
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
            VGPRF_per_pe: 8,
            SGPRF_per_wave: 16,
            wave_size: 8,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
                mov r4.w, lane_id
                utof r4.xyzw, r4.wwww
                mov r4.z, wave_id
                utof r4.z, r4.z
                add_f32 r4.xyzw, r4.xyzw, vec4(0.0 0.0 0.0 1.0)
                lt_f32 r4.xy, r4.ww, vec2(4.0 2.0)
                utof r4.xy, r4.xy
                br_push r4.x, LB_1, LB_2
                mov r0.x, vec1(666.0)
                br_push r4.y, LB_0_1, LB_0_2
                mov r0.y, vec1(666.0)
                pop_mask
            LB_0_1:
                mov r0.y, vec1(777.0)
                pop_mask
            LB_0_2:
                pop_mask
            LB_1:
                mov r0.x, vec1(777.0)

                ; push the current wave mask
                push_mask LOOP_END
            LOOP_PROLOG:
                lt_f32 r4.x, r4.w, vec1(8.0)
                add_f32 r4.w, r4.w, vec1(1.0)
                ; Setting current lane mask
                ; If all lanes are disabled pop_mask is invoked
                ; If mask stack is empty then wave is retired
                mask_nz r4.x
            LOOP_BEGIN:
                jmp LOOP_PROLOG
            LOOP_END:
                pop_mask
                
                
            LB_2:
                mov r4.y, lane_id
                utof r4.y, r4.y
                ret
                ",
            ),
        };
        dispatch(&mut gpu_state, &program, 8, 1);
        let mut mask_history: Vec<Vec<u32>> = Vec::new();
        while clock(&mut gpu_state) {
            for cu in &gpu_state.cus {
                for wave in &cu.waves {
                    if wave.enabled {
                        mask_history.push(
                            wave.exec_mask
                                .iter()
                                .map(|&b| if b { 1 } else { 0 })
                                .collect::<Vec<_>>(),
                        );
                        // print!("{}:\t", wave.pc);
                        // for bit in &wave.exec_mask {
                        //     print!("{}", if *bit { 1 } else { 0 });
                        // }
                        // print!(" ");

                        // println!(
                        //     "vec!{:?},",
                        //     wave.exec_mask
                        //         .iter()
                        //         .map(|&b| if b { 1 } else { 0 })
                        //         .collect::<Vec<_>>()
                        // );
                    }
                }
                // print!(" ");
            }
            // println!("");
        }
        assert_eq!(
            mask_history,
            vec![
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 0, 0, 0, 0, 0],
                vec![1, 1, 1, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 1, 1, 0, 0, 0, 0, 0],
                vec![0, 1, 1, 0, 0, 0, 0, 0],
                vec![1, 1, 1, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![0, 0, 0, 1, 1, 1, 1, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 0],
                vec![0, 0, 0, 1, 1, 1, 0, 0],
                vec![0, 0, 0, 1, 1, 1, 0, 0],
                vec![0, 0, 0, 1, 1, 1, 0, 0],
                vec![0, 0, 0, 1, 1, 1, 0, 0],
                vec![0, 0, 0, 1, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
                vec![1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
    }

    #[test]
    fn alu_overlap_cycles_test() {
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
            VGPRF_per_pe: 8,
            SGPRF_per_wave: 16,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
        };
        // @TODO: 2 cycles are wasted
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
                mov r4.x, vec1(8.0)
                mov r5.y, vec1(2.0)
                mov r0.x, vec1(1.0)
                mov r1.x, vec1(2.0)
                div_f32 r4.x, r5.y, r4.x
                div_f32 r2.x, r0.x, r1.x
                ret
                ",
            ),
        };
        dispatch(&mut gpu_state, &program, 4, 1);
        while clock(&mut gpu_state) {
            // @DEAD
            // for cu in &gpu_state.cus {
            //     for alu in &cu.valus {
            //         print!("{}", if alu.was_busy {
            //             format!("{}", alu.instr.as_ref().unwrap().line)
            //         } else {
            //             String::from("#")
            //         });
            //     }
            //     print!(" ");
            // }
            // println!("");
        }
        assert_eq!(11, gpu_state.clock_counter);
        let mut gpu_state = GPUState::new(&config);
        dispatch(&mut gpu_state, &program, 8, 1);
        while clock(&mut gpu_state) {}
        assert_eq!(14, gpu_state.clock_counter);

        // @DEAD
        // println!("{}", gpu_state.clock_counter);
        // println!(
        //     "{:?}",
        //     gpu_state.cus[0].waves[0].vgprfs[2]
        //         .iter()
        //         .map(|r| castToFValue(&r.val))
        //         .collect::<Vec<_>>()
        // );
    }

    #[test]
    fn test3() {
        let res = parse(
            r"
                mov r4.w, thread_id
                utof r4.xyzw, r4.wwww
                mov r4.z, wave_id
                utof r4.z, r4.z
                add_f32 r4.xyzw, r4.xyzw, vec4(1.0 1.0 0.0 1.0)
                lt_f32 r4.xy, r4.ww, vec2(3.0 2.0)
                utof r4.xy, r4.xy
                br_push r4.x, LB_1, LB_2
                LB_0:
                mov r0.x, vec1(666.0)
                br_push r4.y, LB_0_1, LB_0_2
                LB_0_0:
                mov r0.y, vec1(666.0)
                pop_mask
                LB_0_1:
                mov r0.y, vec1(777.0)
                pop_mask
                LB_0_2:
                pop_mask
                LB_1:
                mov r0.x, vec1(777.0)
                pop_mask
                LB_2:
                mov r4.y, lane_id
                utof r4.y, r4.y
                ret
                ",
        );
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
            VGPRF_per_pe: 8,
            SGPRF_per_wave: 16,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program { ins: res };
        dispatch(&mut gpu_state, &program, 8, 1);
        while clock(&mut gpu_state) {}
        assert_eq!(
            vec![
                [666.0, 666.0, 0.0, 0.0],
                [666.0, 777.0, 0.0, 0.0],
                [777.0, 0.0, 0.0, 0.0],
                [777.0, 0.0, 0.0, 0.0]
            ],
            gpu_state.cus[0].waves[0].vgprfs[0]
                .iter()
                .map(|r| castToFValue(&r.val))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                [0.0, 0.0, 1.0, 5.0],
                [0.0, 1.0, 1.0, 6.0],
                [0.0, 2.0, 1.0, 7.0],
                [0.0, 3.0, 1.0, 8.0]
            ],
            gpu_state.cus[0].waves[1].vgprfs[4]
                .iter()
                .map(|r| castToFValue(&r.val))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test2() {
        let res = parse(
            r"
                mov r1.xyzw, r2.xyzw
                mov r2.x, r3.w
                add_f32 r1.xyzw, r2.xyzw, r3.wzxy
                mov r4.xyzw, vec4 ( 1.0 2.0 3.0 5.0 )
                pop_mask
                ret
                LB_1:
                br_push r1.x, LB_2, LB_3
                LB_2:
                pop_mask
                LB_3:
                pop_mask
                ret
                lt_f32 r1.x, r2.x, r3.y
                mov r4.w, thread_id
                mov r4.w, lane_id
                mov r4.w, wave_id
                utof r4.w, r4.w
                ",
        );
        // println!("{:?}", res);
        assert_eq!(
            res,
            vec![
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 1,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 1,
                            comps: [Component::X, Component::Y, Component::Z, Component::W]
                        }),
                        Operand::VRegister(RegRef {
                            id: 2,
                            comps: [Component::X, Component::Y, Component::Z, Component::W]
                        }),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 2,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 2,
                            comps: [
                                Component::X,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::VRegister(RegRef {
                            id: 3,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::ADD,
                    interp: Interpretation::F32,
                    line: 3,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 1,
                            comps: [Component::X, Component::Y, Component::Z, Component::W]
                        }),
                        Operand::VRegister(RegRef {
                            id: 2,
                            comps: [Component::X, Component::Y, Component::Z, Component::W]
                        }),
                        Operand::VRegister(RegRef {
                            id: 3,
                            comps: [Component::W, Component::Z, Component::X, Component::Y]
                        }),
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 4,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [Component::X, Component::Y, Component::Z, Component::W]
                        }),
                        Operand::Immediate(ImmediateVal::V4F(1.0, 2.0, 3.0, 5.0)),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::POP_MASK,
                    interp: Interpretation::NONE,
                    line: 5,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE,]
                },
                Instruction {
                    ty: InstTy::RET,
                    interp: Interpretation::NONE,
                    line: 6,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE,]
                },
                Instruction {
                    ty: InstTy::BR_PUSH,
                    interp: Interpretation::NONE,
                    line: 8,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 1,
                            comps: [
                                Component::X,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::Label(7),
                        Operand::Label(8),
                        Operand::NONE,
                    ]
                },
                Instruction {
                    ty: InstTy::POP_MASK,
                    interp: Interpretation::NONE,
                    line: 10,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE]
                },
                Instruction {
                    ty: InstTy::POP_MASK,
                    interp: Interpretation::NONE,
                    line: 12,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE]
                },
                Instruction {
                    ty: InstTy::RET,
                    interp: Interpretation::NONE,
                    line: 13,
                    ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE]
                },
                Instruction {
                    ty: InstTy::LT,
                    interp: Interpretation::F32,
                    line: 14,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 1,
                            comps: [
                                Component::X,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::VRegister(RegRef {
                            id: 2,
                            comps: [
                                Component::X,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::VRegister(RegRef {
                            id: 3,
                            comps: [
                                Component::Y,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 15,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::Builtin(BuiltinVal::THREAD_ID),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 16,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::Builtin(BuiltinVal::LANE_ID),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::MOV,
                    interp: Interpretation::NONE,
                    line: 17,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::Builtin(BuiltinVal::WAVE_ID),
                        Operand::NONE,
                        Operand::NONE
                    ]
                },
                Instruction {
                    ty: InstTy::UTOF,
                    interp: Interpretation::NONE,
                    line: 18,
                    ops: [
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::VRegister(RegRef {
                            id: 4,
                            comps: [
                                Component::W,
                                Component::NONE,
                                Component::NONE,
                                Component::NONE
                            ]
                        }),
                        Operand::NONE,
                        Operand::NONE
                    ]
                }
            ]
        );
    }

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
            VGPRF_per_pe: 8,
            SGPRF_per_wave: 16,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let mut program = Program { ins: Vec::new() };
        // 1: mov r1.wzyx, vec4(1.0f, 1.0f, 1.0f, 0.0f)
        // 2: mov r2.wzyx, vec4(1.0f, 2.0f, 3.0f, 4.0f)
        // 3: add_f32 r0.xyzw, r1.xyzw, r2.xyzw
        // 4: mov r1.w, thread_id
        // 5: utof r1.xyzw, r1.wwww
        // 6: add_f32 r3.xyzw, r1.xyzw, r2.xyzw
        // 7: mov r4.w, vec1(777.0f)
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::MOV,
            line: 1,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [Component::W, Component::Z, Component::Y, Component::X],
                }),
                Operand::Immediate(ImmediateVal::V4F(1.0, 1.0, 1.0, 0.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::MOV,
            line: 2,
            ops: [
                Operand::VRegister(RegRef {
                    id: 2,
                    comps: [Component::W, Component::Z, Component::Y, Component::X],
                }),
                Operand::Immediate(ImmediateVal::V4F(1.0, 2.0, 3.0, 4.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
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
        program.ins.push(Instruction {
            interp: Interpretation::NONE,
            ty: InstTy::MOV,
            line: 4,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::W,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Builtin(BuiltinVal::THREAD_ID),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::U32,
            ty: InstTy::UTOF,
            line: 5,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [Component::X, Component::Y, Component::Z, Component::W],
                }),
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [Component::W, Component::W, Component::W, Component::W],
                }),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::ADD,
            line: 6,
            ops: [
                Operand::VRegister(RegRef {
                    id: 3,
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
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::MOV,
            line: 7,
            ops: [
                Operand::VRegister(RegRef {
                    id: 4,
                    comps: [
                        Component::W,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Immediate(ImmediateVal::V1F(777.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });

        // 8: lt_f32 r1.x, r1.y, vec1(2.0f)
        // 9: br_push r1.x, LB_1, LB_3
        //10: LB_0:
        //11: mov r1.x, vec1(1.0f)
        //12: pop
        //13: LB_1:
        //14: mov r1.x, vec1(0.0f)
        //15: pop
        //16: LB_3:
        //17: ret
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::LT,
            line: 8,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::X,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::Y,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Immediate(ImmediateVal::V1F(2.0)),
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::BR_PUSH,
            line: 9,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::X,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Label(11),
                Operand::Label(13),
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::MOV,
            line: 11,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::X,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Immediate(ImmediateVal::V1F(666.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::NONE,
            ty: InstTy::POP_MASK,
            line: 12,
            ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE],
        });
        program.ins.push(Instruction {
            interp: Interpretation::F32,
            ty: InstTy::MOV,
            line: 14,
            ops: [
                Operand::VRegister(RegRef {
                    id: 1,
                    comps: [
                        Component::X,
                        Component::NONE,
                        Component::NONE,
                        Component::NONE,
                    ],
                }),
                Operand::Immediate(ImmediateVal::V1F(777.0)),
                Operand::NONE,
                Operand::NONE,
            ],
        });
        program.ins.push(Instruction {
            interp: Interpretation::NONE,
            ty: InstTy::POP_MASK,
            line: 15,
            ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE],
        });
        program.ins.push(Instruction {
            interp: Interpretation::NONE,
            ty: InstTy::RET,
            line: 17,
            ops: [Operand::NONE, Operand::NONE, Operand::NONE, Operand::NONE],
        });

        // @Test
        dispatch(&mut gpu_state, &program, 4, 1);
        while clock(&mut gpu_state) {}
    }
}
// @Papers
//
// A. Bakhoda, G.L. Yuan, W.W.L. Fung, H. Wong, T.M. Aamodt.
// "Analyzing CUDA workloads using a detailed GPU simulator,"
// Performance Analysis of Systems and Software, 2009. ISPASS 2009.
//
//
// @TODOLIST
// * shuffle instructions
// * memory io instructions
//     * buffer/texture binding mechanism
//     * cache system
//     * sampling system
//     * atomic operations
// * thread group instructions
//     * barriers
//     * fences
//     * SLM
//     * atomics
//
