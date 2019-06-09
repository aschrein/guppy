//#![feature(proc_macro, wasm_custom_section, wasm_import_module)]
#![feature(generators, generator_trait)]
use std::collections::*;

type Value = [u32; 4];
type FValue = [f32; 4];

#[derive(Clone, Copy, Debug)]
struct Register {
    val: Value,
    // Waits for an operation to WB
    locked: bool,
    // As we buffer input operands we don't need to stall on WAR hazards
}

fn applyReadSwizzle(val: &Value, comps: &[Component; 4]) -> Value {
    let mut out: Value = [0, 0, 0, 0];
    for i in 0..4 {
        match comps[i] {
            Component::X => out[i] = val[0],
            Component::Y => out[i] = val[1],
            Component::Z => out[i] = val[2],
            Component::W => out[i] = val[3],
            Component::NONE => {}
        }
    }
    out
}
fn applyWriteSwizzle(out: &mut Value, val: &Value, comps: &[Component; 4]) {
    let mut out_: Value = *out;
    for i in 0..4 {
        match comps[i] {
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
    let mut out: Value = [0, 0, 0, 0];
    for i in 0..4 {
        out[i] = if that[i] != 0 {
            this[i] / that[i]
        } else {
            0xffffffff
        };
    }
    out
}

fn LTU32(this: &Value, that: &Value) -> Value {
    [
        if this[0] < that[0] { 1 } else { 0 },
        if this[1] < that[1] { 1 } else { 0 },
        if this[2] < that[2] { 1 } else { 0 },
        if this[3] < that[3] { 1 } else { 0 },
    ]
}

fn ORU32(this: &Value, that: &Value) -> Value {
    [
        this[0] | that[0],
        this[1] | that[1],
        this[2] | that[2],
        this[3] | that[3],
    ]
}

fn ANDU32(this: &Value, that: &Value) -> Value {
    [
        this[0] & that[0],
        this[1] & that[1],
        this[2] & that[2],
        this[3] & that[3],
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
    let mut out: FValue = [0.0, 0.0, 0.0, 0.0];
    for i in 0..4 {
        // Handle div by zero properly for fucks sake
        out[i] = if that[i] != 0.0 {
            this[i] / that[i]
        } else {
            1.0e23
        };
    }
    castToValue(&out)
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

#[derive(Clone, Debug)]
enum View {
    BUFFER(Buffer),
    TEXTURE2D(Texture2D),
    NONE,
}

struct WaveState {
    // t0..t63 registers
    r_views: Vec<View>,
    // u0..u63 registers
    rw_views: Vec<View>,
    // s0..s63 registers
    samplers: Vec<Sampler>,
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
    clock_counter: u32,
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
            r_views: Vec::new(),
            rw_views: Vec::new(),
            samplers: Vec::new(),
            vgprfs: Vec::new(),
            sgprf: Vec::new(),
            wave_id: 0,
            group_id: 0,
            group_size: 0,
            program: None,
            pc: 0,
            clock_counter: 0,
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
                .map(|r| applyReadSwizzle(&r.val, &vop.comps))
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
        r_views: &Vec<View>,
        rw_views: &Vec<View>,
        samplers: &Vec<Sampler>,
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
        // for i in 0..config.SGPRF_per_wave {
        //     sgprf.push(Register {
        //         val: [0, 0, 0, 0],
        //         locked: false,
        //     });
        // }
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
        self.clock_counter = 0;
        self.wave_id = wave_id;
        self.group_id = group_id;
        self.group_size = group_size;
        self.has_been_dispatched = false;
        self.enabled = true;
        self.r_views = r_views.to_vec().clone();
        self.rw_views = rw_views.to_vec().clone();
        self.samplers = samplers.to_vec().clone();
    }
}

#[derive(Clone, Debug)]
// Dispatched instruction
struct DispInstruction {
    // Src ops buffer to avoid WAR hazards
    // 3 is the max number of input ops per instruction
    src: [Option<Vec<Value>>; 3],
    instr: Option<Instruction>,
    exec_mask: Option<Vec<bool>>,
    wave_id: u32,
    timer: u32,
}

struct VALUState {
    active: bool,
    // Size is fixed
    pipe: Vec<Option<DispInstruction>>,
}

impl VALUState {
    fn pop(&mut self) -> Option<DispInstruction> {
        // Decrease the counter on those instructions
        for item in &mut self.pipe {
            if item.is_some() && item.as_ref().unwrap().timer > 0 {
                item.as_mut().unwrap().timer -= 1;
            }
        }
        // The last instruction stalls the pipe
        if self.pipe.last().unwrap().is_some()
            && self.pipe.last().unwrap().as_ref().unwrap().timer != 0
        {
            return None;
        }
        // Pop the last and shift the rest
        let last = self.pipe.pop().unwrap();
        // @TODO: There must be a better way to do this shift-register type of thing
        let mut new_pipe: Vec<Option<DispInstruction>> = Vec::new();
        new_pipe.push(None);
        for item in &self.pipe {
            if item.is_some() {
                new_pipe.push(Some(item.as_ref().unwrap().clone()));
            } else {
                new_pipe.push(None);
            }
        }
        self.pipe = new_pipe;
        last
    }
    fn ready(&self) -> bool {
        self.pipe.first().unwrap().is_none()
    }
    fn push(&mut self, inst: &DispInstruction) -> bool {
        if self.pipe.first().unwrap().is_none() {
            self.pipe[0] = Some(inst.clone());
            return true;
        }
        false
    }
}

struct SALUState {
    // @TODO
// instr: Option<Instruction>,
// wave_id: u32,
// timer: u32,
// busy: bool,
}

#[derive(Clone, Debug)]
struct SampleReq {
    cu_id: u32,
    wave_id: u32,
    reg_row: u32,
    reg_col: u32,
    read_comps: [Component; 4],
    write_comps: [Component; 4],
    u: f32,
    v: f32,
    texture: u32,
    sampler: u32,
}

#[derive(Clone, Debug)]
struct SampleReqWrap {
    req: SampleReq,
    // buffer for (mem_offset, value)[4]
    // (-1, -1) (1, -1) (1 1) (-1 1)
    values: [(u32, f32, Option<u32>); 4],
    timer: u32,
}

#[derive(Clone, Debug)]
struct SamplerState {
    cache_table: CacheTable,
    reqs: Vec<Option<SampleReqWrap>>,
    free_reqs: Vec<u32>,
    wait_reqs: HashSet<u32>,
    // mem_offset -> [req_id]
    // coalesced request table
    gather_queue: HashMap<u32, HashSet<u32>>,
}

impl SamplerState {
    fn new(gpu_config: &GPUConfig) -> SamplerState {
        let mut reqs: Vec<Option<SampleReqWrap>> = Vec::new();
        for i in 0..1024 {
            reqs.push(None);
        }
        let mut free_reqs: Vec<u32> = Vec::new();
        for i in 0..1024 {
            free_reqs.push(i);
        }
        SamplerState {
            cache_table: CacheTable::new(gpu_config.sampler_cache_size / 64, 2),
            reqs: reqs,
            free_reqs: free_reqs,
            wait_reqs: HashSet::new(),
            gather_queue: HashMap::new(),
        }
    }
    fn alloc_id(&mut self) -> Option<u32> {
        if self.free_reqs.len() != 0 {
            let id = self.free_reqs.pop().unwrap();
            self.wait_reqs.insert(id);
            Some(id)
        } else {
            None
        }
    }
    fn get_free_slots(&self) -> u32 {
        self.free_reqs.len() as u32
    }
}

#[derive(Clone, Debug)]
// 4 byte LD generated by a lane
struct LDReq {
    reg_row: u32,
    reg_col: u32,
    mem_offset: u32,
    timer: u32,
}

#[derive(Clone, Debug)]
struct CacheLine {
    address: u32,
    mem: [u32; 16],
}

#[derive(Clone, Debug)]
struct CacheTable {
    // tag -> [(address, cache_line)]
    // mind the associativity
    // the last line in the bin is the newest
    contents: Vec<Vec<Option<CacheLine>>>,
    associativity: u32,
    size: u32,
}

impl CacheTable {
    fn new(size: u32, associativity: u32) -> CacheTable {
        let mut contents: Vec<Vec<Option<CacheLine>>> = Vec::new();
        for i in 0..size {
            let mut bin: Vec<Option<CacheLine>> = Vec::new();
            for j in 0..associativity {
                bin.push(None)
            }
            contents.push(bin);
        }
        CacheTable {
            associativity: associativity,
            size: size,
            contents: contents,
        }
    }
    fn getIndex(&self, mem_offset: u32) -> u32 {
        // Calculate the bin tag/index of the cacheline
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        mem_offset.hash(&mut s);
        (s.finish() % (self.size as u64)) as u32
    }
    fn putLine(&mut self, line: &CacheLine) {
        let index = self.getIndex(line.address);
        let mut new_bin: Vec<Option<CacheLine>> = Vec::new();
        // evict the eldest line
        for line in &self.contents[index as usize][1..] {
            if let Some(line_some) = line {
                new_bin.push(Some((*line_some).clone()));
            } else {
                new_bin.push(None);
            }
        }
        new_bin.push(Some((*line).clone()));
        self.contents[index as usize] = new_bin;
    }
    fn getLine(&self, page_offset: u32) -> Option<CacheLine> {
        let index = self.getIndex(page_offset);
        for line in &self.contents[index as usize] {
            if let Some(line) = line {
                if line.address == page_offset {
                    return Some(line.clone());
                }
            }
        }
        None
    }
}

struct L1CacheState {
    cache_table: CacheTable,
    // mem_offset -> wave_id -> [waiting_registers]
    // coalesced request table
    gather_queue: HashMap<u32, HashMap<u32, Vec<LDReq>>>,
}

#[derive(Hash, Debug, Clone)]
struct L2PageReq {
    l1_target: bool,
    timer: u32,
}

struct L2CacheState {
    cache_table: CacheTable,
    // mem_offset -> cu_id -> timeout
    // @TODO: limit the maximum number of request queue
    req_queue: HashMap<u32, HashMap<u32, L2PageReq>>,
    // smp_queue: HashMap<u32, HashMap<u32, u32>>,
}

impl L1CacheState {
    fn new(size: u32, associativity: u32) -> L1CacheState {
        L1CacheState {
            cache_table: CacheTable::new(size, associativity),
            gather_queue: HashMap::new(),
        }
    }
}

impl L2CacheState {
    fn new(size: u32, associativity: u32) -> L2CacheState {
        L2CacheState {
            cache_table: CacheTable::new(size, associativity),
            req_queue: HashMap::new(),
        }
    }
}

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
    sampler: SamplerState,
    l1: L1CacheState,
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
            let mut pipe: Vec<Option<DispInstruction>> = Vec::new();
            for i in 0..config.ALU_pipe_len {
                pipe.push(None);
            }
            valus.push(VALUState {
                active: false,
                pipe: pipe,
            });
        }
        salus.push(SALUState {});
        let mut fes: Vec<FEState> = Vec::new();
        for i in 0..config.fd_per_cu {
            fes.push(FEState {});
        }
        CUState {
            waves: waves,
            valus: valus,
            salus: salus,
            fes: fes,
            sampler: SamplerState::new(&config),
            l1: L1CacheState::new(config.L1_size / 64, 2),
            slm: SLMState {},
        }
    }
}

struct GPUState {
    // ongoing work
    cus: Vec<CUState>,
    clock_counter: u32,
    l2: L2CacheState,
    // queue for future work
    dreqs: Vec<DispatchReq>,
    // Well, memory
    mem: Vec<u32>,
    // page_offset -> timer
    ld_reqs: HashMap<u32, u32>,
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
            l2: L2CacheState::new(config.L2_size / 64, 2),
            dreqs: Vec::new(),
            config: config.clone(),
            mem: Vec::new(),
            clock_counter: 0,
            ld_reqs: HashMap::new(),
        }
    }
    fn get_alu_active(&self) -> f64 {
        let mut res: u32 = 0;
        for cu in &self.cus {
            for alu in &cu.valus {
                if alu.active {
                    res += 1;
                }
            }
        }
        100.0 * (res as f64) / (self.config.CU_count * self.config.ALU_per_cu) as f64
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
    fn serve_sample(&mut self, cu_id: u32, req_id: u32) {
        let req = self.cus[cu_id as usize].sampler.reqs[req_id as usize]
            .clone()
            .unwrap();
        self.cus[cu_id as usize].sampler.reqs[req_id as usize] = None;
        self.cus[cu_id as usize].sampler.free_reqs.push(req_id);
        // let mut final_val = 0.0;
        let mut final_val: FValue = [0.0, 0.0, 0.0, 0.0];
        let texture = match &self.cus[req.req.cu_id as usize].waves[req.req.wave_id as usize]
            .r_views[req.req.texture as usize]
        {
            View::TEXTURE2D(tex) => tex.clone(),
            _ => std::panic!(),
        };

        for i in 0..4 {
            for j in 0..4 {
                final_val[j] += match texture.format {
                    TextureFormat::RGBA8_UNORM => {
                        ((req.values[i].2.unwrap() >> ((3 - j) * 8)) & 0xff) as f32 / 255.0
                    }
                    _ => std::panic!(""),
                } * req.values[i].1;
            }
        }
        let reg = &mut self.cus[cu_id as usize].waves[req.req.wave_id as usize].vgprfs
            [req.req.reg_row as usize][req.req.reg_col as usize];
        reg.locked = false;
        let u32val = castToValue(&final_val);
        applyWriteSwizzle(
            &mut reg.val,
            &applyReadSwizzle(&u32val, &req.req.read_comps),
            &req.req.write_comps,
        );
    }
    fn sampler_put_line(&mut self, cu_id: u32, line: &CacheLine) {
        let requests = self.cus[cu_id as usize]
            .sampler
            .gather_queue
            .remove(&line.address);
        // the line is in the cache so serve the requiests
        if let Some(requests) = requests {
            // Serve the request
            for requests_id in requests {
                let mut req = self.cus[cu_id as usize].sampler.reqs[requests_id as usize]
                    .clone()
                    .unwrap();
                assert_eq!(req.timer, 0);
                for val in &mut req.values {
                    let page_offset = val.0 & !63;
                    if line.address == page_offset {
                        val.2 = Some(line.mem[((val.0 - line.address) / 4) as usize]);
                    }
                }
                let complete = req.values.iter().all(|x| x.2.is_some());
                self.cus[cu_id as usize].sampler.reqs[requests_id as usize] = Some(req);
                // The request is ready
                if complete {
                    self.serve_sample(cu_id, requests_id);
                }
            }
        }
        self.cus[cu_id as usize].sampler.cache_table.putLine(&line);
    }
    fn sample(&mut self, req: &SampleReq) -> bool {
        // 1) required texels for bilinear interpl
        // 2) weights
        // 3) pages
        // 4) assemble the request
        let mut texels: Vec<(f32, i32, i32, u32)> = Vec::new();
        let sampler = self.cus[req.cu_id as usize].waves[req.wave_id as usize].samplers
            [req.sampler as usize]
            .clone();
        let texture = match &self.cus[req.cu_id as usize].waves[req.wave_id as usize].r_views
            [req.texture as usize]
        {
            View::TEXTURE2D(tex) => tex.clone(),
            _ => std::panic!(),
        };
        let uv = match sampler.wrap_mode {
            WrapMode::WRAP => {
                let mut u = req.u;
                let mut v = req.v;
                while u > 1.0 {
                    u -= 1.0
                }
                while u < 0.0 {
                    u += 1.0
                }
                while v > 1.0 {
                    v -= 1.0
                }
                while v < 0.0 {
                    v += 1.0
                }

                (req.u * texture.width as f32, req.v * texture.height as f32)
            }
            _ => std::panic!(""),
        };
        let floor = |x: f32| {
            if x < 0.0 {
                (x - 1.0) as i32
            } else {
                x as i32
            }
        };
        // Transfrom texture space into texel space
        // (0.5, 0.5) - texture coordinate of the first texel
        texels.push((0.0, floor(uv.0 - 0.5), floor(uv.1 - 0.5), 0));
        texels.push((0.0, floor(uv.0 + 0.5), floor(uv.1 - 0.5), 0));
        texels.push((0.0, floor(uv.0 + 0.5), floor(uv.1 + 0.5), 0));
        texels.push((0.0, floor(uv.0 - 0.5), floor(uv.1 + 0.5), 0));
        // Figure out interpolation weights
        for tex in &mut texels {
            let tu = 1.0 - f32::abs(uv.0 - 0.5 - (tex.1 as f32));
            let tv = 1.0 - f32::abs(uv.1 - 0.5 - (tex.2 as f32));
            let weight = tu * tv;
            tex.0 = weight;
        }
        // Allocate the request id
        let req_id = self.cus[req.cu_id as usize].sampler.alloc_id();
        if req_id.is_none() {
            return false;
        }
        let req_id = req_id.unwrap();
        // Calculate needed pages
        for tex in &mut texels {
            let texel_coord = match sampler.wrap_mode {
                WrapMode::WRAP => {
                    //tex.1 % req.texture.width, tex.2 % req.texture.height),
                    let mut ntex = (tex.1, tex.2);
                    while ntex.0 >= texture.width as i32 {
                        ntex.0 -= texture.width as i32;
                    }
                    while ntex.0 < 0 {
                        ntex.0 += texture.width as i32;
                    }
                    while ntex.1 >= texture.height as i32 {
                        ntex.1 -= texture.height as i32;
                    }

                    while ntex.1 < 0 {
                        ntex.1 += texture.height as i32;
                    }
                    (ntex.0 as u32, ntex.1 as u32)
                }
                _ => std::panic!(""),
            };
            let mem_offset = match texture.format {
                TextureFormat::RGBA8_UNORM => {
                    texture.offset + texture.pitch * texel_coord.1 + texel_coord.0 * 4 // 4 bytes per pixel
                }
                _ => std::panic!(""),
            };
            tex.3 = mem_offset;
        }
        let timer = self.config.sampler_latency;
        self.cus[req.cu_id as usize].sampler.reqs[req_id as usize] = Some(SampleReqWrap {
            req: req.clone(),
            values: [
                (texels[0].3, texels[0].0, None),
                (texels[1].3, texels[1].0, None),
                (texels[2].3, texels[2].0, None),
                (texels[3].3, texels[3].0, None),
            ],
            timer: timer,
        });
        true
    }
    fn sampler_clock(&mut self, cu_id: u32) {
        let mut new_ready_reqs: Vec<u32> = Vec::new();
        let wait_reqs = self.cus[cu_id as usize].sampler.wait_reqs.clone();
        // split lists
        for &req_id in &wait_reqs {
            let req = self.cus[cu_id as usize].sampler.reqs[req_id as usize]
                .as_mut()
                .unwrap();
            if req.timer == 1 {
                new_ready_reqs.push(req_id);
            }
            req.timer = if req.timer != 0 { req.timer - 1 } else { 0 };
        }

        let mut missed_pages: Vec<(u32, u32)> = Vec::new();
        // Try to serve ready requests
        for req_id in new_ready_reqs {
            self.cus[cu_id as usize].sampler.wait_reqs.remove(&req_id);
            let mut req = self.cus[cu_id as usize].sampler.reqs[req_id as usize]
                .clone()
                .unwrap();
            assert_eq!(req.timer, 0);
            // Each value
            for val in &mut req.values {
                assert!(val.2.is_none());
                let page_offset = val.0 & !63;
                if let Some(line) = self.cus[cu_id as usize]
                    .sampler
                    .cache_table
                    .getLine(page_offset)
                {
                    val.2 = Some(line.mem[((val.0 - page_offset) / 4) as usize]);
                } else {
                    missed_pages.push((page_offset, req_id));
                }
            }
            let complete = req.values.iter().all(|x| x.2.is_some());
            self.cus[cu_id as usize].sampler.reqs[req_id as usize] = Some(req);
            if complete {
                self.serve_sample(cu_id, req_id);
            }
        }
        // For each missed page
        for (page, req_id) in missed_pages {
            // Request the page from l2
            self.l2_request_page(cu_id, page, false);
            // Register the page in the request table
            let queue = &mut self.cus[cu_id as usize].sampler.gather_queue;
            if !queue.contains_key(&page) {
                queue.insert(page, HashSet::new());
            }
            queue.get_mut(&page).unwrap().insert(req_id);
        }
    }
    fn loadMem(&mut self, page_offset: u32, timeout: u32) {
        let page_size = 64 as u32;
        assert!((page_offset & (page_size - 1)) == 0);
        if !self.ld_reqs.contains_key(&page_offset) {
            self.ld_reqs.insert(page_offset, timeout);
        }
    }
    fn wave_gather(&mut self, cu_id: u32, wave_id: u32, reqs: &Vec<LDReq>) {
        let page_size = 64 as u32;
        for req in reqs {
            {
                let l1 = &mut self.cus[cu_id as usize].l1;
                let page_offset = req.mem_offset & !(page_size - 1);
                // Boilerplate to append the entry to the table
                if !l1.gather_queue.contains_key(&page_offset) {
                    l1.gather_queue.insert(page_offset, HashMap::new());
                }
                if !l1
                    .gather_queue
                    .get(&page_offset)
                    .unwrap()
                    .contains_key(&wave_id)
                {
                    l1.gather_queue
                        .get_mut(&page_offset)
                        .unwrap()
                        .insert(wave_id, Vec::new());
                }
                // EOF Boilerplate
                // Put the requests into the queue
                // @ASSUME: requests are unique as long as registers are locked
                l1.gather_queue
                    .get_mut(&page_offset)
                    .unwrap()
                    .get_mut(&wave_id)
                    .unwrap()
                    .push((*req).clone());
            }
            // Lock the register
            self.cus[cu_id as usize].waves[wave_id as usize].vgprfs[req.reg_row as usize]
                [(req.reg_col / 4) as usize]
                .locked = true;
        }
    }
    fn l1_clock(&mut self, cu_id: u32) {
        // Decrement timers and split the requests into un/ready lists
        let mut ready_reqs: Vec<(u32, u32, LDReq)> = Vec::new();
        let mut new_gather_queue: HashMap<u32, HashMap<u32, Vec<LDReq>>> = HashMap::new();
        {
            let cu_state = &self.cus[cu_id as usize];
            for (&mem_offset, reqq) in &cu_state.l1.gather_queue {
                let mut new_wave_q: HashMap<u32, Vec<LDReq>> = HashMap::new();
                for (&wave_id, req) in reqq {
                    let mut new_reqs: Vec<LDReq> = Vec::new();
                    for ldreq in req {
                        let mut new_ldreq = ldreq.clone();
                        new_ldreq.timer = if ldreq.timer > 0 { ldreq.timer - 1 } else { 0 };
                        if new_ldreq.timer == 0 {
                            ready_reqs.push((mem_offset, wave_id, new_ldreq.clone()));
                        } else {
                            new_reqs.push(new_ldreq);
                        }
                    }
                    new_wave_q.insert(wave_id, new_reqs);
                }
                new_gather_queue.insert(mem_offset, new_wave_q);
            }
        }
        // Try to serve the requests
        for (mem_offset, wave_id, req) in ready_reqs {
            let mut served = false;
            // Try to find the page in the table
            {
                let cu_state = &mut self.cus[cu_id as usize];
                for bin in &cu_state.l1.cache_table.contents {
                    for line in bin {
                        if let Some(line) = line {
                            if line.address == mem_offset {
                                // Cache hit
                                let wave = &mut cu_state.waves[wave_id as usize];
                                // @COPYPASTE
                                assert!(req.mem_offset >= line.address);
                                let reg = &mut wave.vgprfs[req.reg_row as usize]
                                    [(req.reg_col / 4) as usize];
                                reg.val[(req.reg_col % 4) as usize] =
                                    line.mem[((req.mem_offset - line.address) / 4) as usize];
                                // @REFINE: Unlock the register. Assuming there could be no other
                                // pending requests for this register.
                                reg.locked = false;
                                served = true;
                            }
                        }
                    }
                }
            }
            // Cache miss
            if !served {
                // Emit the request to L2 cache
                self.l2_request_page(cu_id, mem_offset, true);
                // Put the request back to the queue
                new_gather_queue
                    .get_mut(&mem_offset)
                    .unwrap()
                    .get_mut(&wave_id)
                    .unwrap()
                    .push(req.clone());
            }
        }
        let mut cu_state = &mut self.cus[cu_id as usize];
        cu_state.l1.gather_queue = new_gather_queue;
    }
    fn l1_put_line(&mut self, cu_id: u32, line: &CacheLine) {
        let cu_state = &mut self.cus[cu_id as usize];
        // the line is in the cache so serve the requiests
        if cu_state.l1.gather_queue.contains_key(&line.address) {
            let mut unserved: Vec<(u32, Vec<LDReq>)> = Vec::new();
            // Serve the request
            for (&wave_id, requests) in cu_state.l1.gather_queue.get(&line.address).unwrap() {
                let wave = &mut cu_state.waves[wave_id as usize];
                let mut unserved_reqs: Vec<LDReq> = Vec::new();
                for req in requests {
                    // Isn't done waiting yet
                    if req.timer != 0 {
                        unserved_reqs.push(req.clone());
                        continue;
                    }
                    // @COPYPASTE
                    assert!(req.mem_offset >= line.address);
                    let reg = &mut wave.vgprfs[req.reg_row as usize][(req.reg_col / 4) as usize];
                    reg.val[(req.reg_col % 4) as usize] =
                        line.mem[((req.mem_offset - line.address) / 4) as usize];
                    // @REFINE: Unlock the register. Assuming there could be no other
                    // pending requests for this register.
                    reg.locked = false;
                }
                if unserved_reqs.len() != 0 {
                    unserved.push((wave_id, unserved_reqs));
                }
            }

            // Remove the request
            if unserved.len() != 0 {
                cu_state
                    .l1
                    .gather_queue
                    .insert(line.address, HashMap::new());
                for (wave_id, requests) in &unserved {
                    cu_state
                        .l1
                        .gather_queue
                        .get_mut(&line.address)
                        .unwrap()
                        .insert(*wave_id, requests.clone());
                }
            } else {
                cu_state.l1.gather_queue.remove_entry(&line.address);
            }
        }
        cu_state.l1.cache_table.putLine(&line);
    }

    fn l2_request_page(&mut self, cu_id: u32, page_offset: u32, l1_target: bool) {
        let page_size = 64 as u32;
        assert!(((page_size as i32) & -(page_size as i32)) == (page_size as i32));
        assert!((page_offset & (page_size - 1)) == 0);
        if !self.l2.req_queue.contains_key(&page_offset) {
            self.l2.req_queue.insert(page_offset, HashMap::new());
        }
        if !self
            .l2
            .req_queue
            .get(&page_offset)
            .unwrap()
            .contains_key(&cu_id)
        {
            self.l2.req_queue.get_mut(&page_offset).unwrap().insert(
                cu_id,
                L2PageReq {
                    timer: self.config.L2_latency,
                    l1_target: l1_target,
                },
            );
        }
    }

    fn l2_clock(&mut self) {
        // As usual, decrement counters and split lists
        let mut ready_reqs: Vec<(u32, u32, L2PageReq)> = Vec::new();
        let mut new_req_queue: HashMap<u32, HashMap<u32, L2PageReq>> = HashMap::new();
        for (&mem_offset, reqq) in &self.l2.req_queue {
            let mut new_cu_q: HashMap<u32, L2PageReq> = HashMap::new();
            for (&cu_id, req) in reqq {
                let mut new_req = req.clone();
                new_req.timer = if req.timer > 0 { req.timer - 1 } else { 0 };
                if new_req.timer == 0 {
                    ready_reqs.push((cu_id, mem_offset, new_req.clone()));
                } else {
                    new_cu_q.insert(cu_id, new_req.clone());
                }
            }
            if !new_cu_q.is_empty() {
                new_req_queue.insert(mem_offset, new_cu_q);
            }
        }
        // Try to serve on ready requests
        for (cu_id, mem_offset, req) in ready_reqs {
            let mut hit_line: Option<CacheLine> = None;
            // Try to find the page in the table
            for bin in &mut self.l2.cache_table.contents {
                for line in bin {
                    if let Some(line) = line {
                        if line.address == mem_offset {
                            // Cache hit
                            hit_line = Some(line.clone());
                        }
                    }
                }
            }
            // Cache hit
            if let Some(line) = hit_line {
                if req.l1_target {
                    self.l1_put_line(cu_id, &line);
                } else {
                    self.sampler_put_line(cu_id, &line);
                }
            }
            // Cache miss
            else {
                // Emit the request to memory
                self.loadMem(mem_offset, self.config.DRAM_latency);
                // Put the request back to the queue
                if !new_req_queue.contains_key(&mem_offset) {
                    new_req_queue.insert(mem_offset, HashMap::new());
                }
                new_req_queue
                    .get_mut(&mem_offset)
                    .unwrap()
                    .insert(cu_id, req.clone());
            }
        }
        self.l2.req_queue = new_req_queue;
    }
    fn l2_put_line(&mut self, line: &CacheLine) {
        // the line is in the cache so serve the requiests
        if self.l2.req_queue.contains_key(&line.address) {
            // Serve the request
            let reqs = self.l2.req_queue.remove_entry(&line.address).unwrap().1;
            for (&cu_id, req) in &reqs {
                if req.timer != 0 {
                    continue;
                }
                if req.l1_target {
                    self.l1_put_line(cu_id, &line);
                } else {
                    self.sampler_put_line(cu_id, &line);
                }
            }
        }
        self.l2.cache_table.putLine(&line);
    }
    fn mem_clock(&mut self) {
        let mut ready_reqs: Vec<u32> = Vec::new();
        let mut new_req_queue: HashMap<u32, u32> = HashMap::new();
        for (&mem_offset, &timer) in &self.ld_reqs {
            let new_timer = if timer > 0 { timer - 1 } else { 0 };
            if new_timer == 0 {
                ready_reqs.push(mem_offset);
            } else {
                new_req_queue.insert(mem_offset, new_timer);
            }
        }
        let mut delivered = 0 as u32;
        for mem_offset in ready_reqs {
            // Limit the bandwidth
            if delivered >= self.config.DRAM_bandwidth {
                new_req_queue.insert(mem_offset, 0);
            }
            let mut line = CacheLine {
                address: mem_offset,
                mem: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            };
            for i in 0..16 {
                line.mem[i] = self.mem[(mem_offset as usize / 4 + i) as usize];
            }
            self.l2_put_line(&line);
            delivered += 64;
        }
        self.ld_reqs = new_req_queue;
    }
}

struct ALUInstMeta {
    latency: u32,
    throughput: u32,
}

#[macro_use]
extern crate serde;
extern crate serde_json;

use serde::{Deserialize, Serialize};
use serde_json::Result;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GPUConfig {
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
    // sampler_cache_latency: u32,
    // Latency of sampler pipeline w/o memory latency
    sampler_latency: u32,

    // Shared Local Memory is part of CU
    // ~16kb
    // SLM_size: u32,
    // ~20clk
    // SLM_latency: u32,
    // ~32
    // SLM_banks: u32,

    // Number of [u32 X 4] per Processing Unit
    VGPRF_per_pe: u32,
    // Number of [u32 X 4] per CU
    // SGPRF_per_wave: u32,

    // SIMD width of ALU
    wave_size: u32,
    // Compute Unit count per GPU
    CU_count: u32,
    ALU_per_cu: u32,
    // Number of SIMD threads which execution is handled by CU
    waves_per_cu: u32,
    // Number of fetch decode units per CU
    fd_per_cu: u32,
    ALU_pipe_len: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum Component {
    X,
    Y,
    Z,
    W,
    NONE,
}

#[derive(Clone, Debug)]
struct Buffer {
    offset: u32,
    size: u32,
}

#[derive(Clone, Debug)]
enum TextureFormat {
    RGBA8_UNORM,
    RGB8_UNORM,
}

#[derive(Clone, Debug)]
struct Texture2D {
    offset: u32,
    pitch: u32,
    height: u32,
    width: u32,
    format: TextureFormat,
}

#[derive(Clone, Debug)]
enum WrapMode {
    CLAMP,
    WRAP,
}

#[derive(Clone, Debug)]
enum SampleMode {
    POINT,
    BILINEAR,
}

// enum SampleFormat {
//     NORM,
//     RAW,
// }

#[derive(Clone, Debug)]
struct Sampler {
    wrap_mode: WrapMode,
    sample_mode: SampleMode,
    // sample_format: SampleFormat,
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
struct RMemRef {
    id: u32,
    comps: [Component; 4],
}

#[derive(Debug, Clone, PartialEq)]
struct RWMemRef {
    id: u32,
    comps: [Component; 4],
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
    RMemory(RMemRef),
    RWMemory(RWMemRef),
    Immediate(ImmediateVal),
    Builtin(BuiltinVal),
    Label(u32),
    Sampler(u32),
    NONE,
}

#[derive(Debug, Clone, PartialEq)]
enum InstTy {
    MOV,
    ADD,
    SUB,
    MUL,
    DIV,
    AND,
    OR,
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
    fn assertFourOp(&self) {
        for op in &self.ops {
            match op {
                Operand::NONE => {
                    std::panic!("");
                }
                _ => {}
            }
        }
    }
}

//use std::ops::Generator;

enum CmdStatus {
    YIELD,
    COMPLETE,
}

type InstFn = Fn(&mut GPUState) -> CmdStatus;

// Generator function for an instruction
type InstGen = Vec<Box<InstFn>>;

// fn matchInst(inst: &Instruction) -> InstGen {
//     let pause = |time: u32| -> Box<InstFn> {
//         let mut pauseTimer = time;
//         Box::new(|gpustate: &mut GPUState| {
//             pauseTimer -= 1;
//             if pauseTimer == 0 {
//                 CmdStatus::COMPLETE
//             } else {
//                 CmdStatus::YIELD
//             }
//         })
//     };
//     let mut out: InstGen = Vec::new();
//     match &inst.ty {
//         InstTy::MOV => {
//             // out.push(Box::new(
//             //     |gpustate: &mut GPUState| {

//             //     }
//             // ))
//         }
//         _ => std::panic!(),
//     };
//     out
// }

#[derive(Clone, Debug)]
struct Program {
    ins: Vec<Instruction>,
}

struct DispatchReq {
    program: Rc<Program>,
    r_views: Vec<View>,
    rw_views: Vec<View>,
    samplers: Vec<Sampler>,
    group_size: u32,
    group_id: u32,
}

fn lower_bound<'a, K: std::cmp::Ord, V>(tree: &'a BTreeMap<K, V>, val: &K) -> (&'a K, &'a V) {
    use std::ops::Bound::*;

    let mut before = tree.range((Unbounded, Excluded(val)));

    (before.next_back().unwrap().0, before.next_back().unwrap().1)
}

fn dispatch(
    gpu_state: &mut GPUState,
    program: &Program,
    r_views: Vec<View>,
    rw_views: Vec<View>,
    samplers: Vec<Sampler>,
    group_size: u32,
    group_count: u32,
) {
    // Check buffer aliasing
    {
        let mut tree: BTreeMap<u32, u32> = BTreeMap::new();
        for view in &r_views {
            match view {
                View::BUFFER(buf) => {
                    assert!(tree.insert(buf.offset, buf.size).is_none());
                }
                View::TEXTURE2D(tex) => {
                    assert!(tree.insert(tex.offset, tex.pitch * tex.height).is_none());
                }
                _ => std::panic!(""),
            }
        }
        for view in &rw_views {
            match view {
                View::BUFFER(buf) => {
                    assert!(tree.insert(buf.offset, buf.size).is_none());
                }
                View::TEXTURE2D(tex) => {
                    assert!(tree.insert(tex.offset, tex.pitch * tex.height).is_none());
                }
                _ => std::panic!(""),
            }
        }
        let mut prev_back: Option<u32> = None;
        for (&offset, &size) in &tree {
            if let Some(prev_back) = prev_back {
                assert!(prev_back <= offset);
            }
            prev_back = Some(offset + size);
        }
    }
    for g_id in 0..group_count {
        let disp_req = DispatchReq {
            program: Rc::new(program.clone()),
            group_size: group_size,
            group_id: g_id,
            r_views: r_views.clone(),
            rw_views: rw_views.clone(),
            samplers: samplers.clone(),
        };
        gpu_state.dreqs.push(disp_req);
    }
}

fn clock(gpu_state: &mut GPUState) -> Option<Vec<Event>> {
    let mut events: Vec<Event> = Vec::new();
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
                            && req.group_size != 0
                    );

                    // @DEAD
                    // assert!(req.group_count * req.group_size % gpu_state.config.wave_size == 0);

                    let warp_count = req.group_size / gpu_state.config.wave_size;
                    if warp_count > cnt_free_waves {
                        deferredReq.push(req);
                        continue;
                    }
                    cnt_free_waves -= warp_count;
                    // Find a free wave and dispatch on it
                    for wave_id in 0..((req.group_size + gpu_state.config.wave_size - 1)
                        / gpu_state.config.wave_size)
                    {
                        let wave_path = gpu_state.findFreeWave().unwrap();
                        let config = &gpu_state.config;
                        gpu_state.cus[wave_path.0].waves[wave_path.1].dispatch(
                            config,
                            &req.program,
                            &req.r_views,
                            &req.rw_views,
                            &req.samplers,
                            wave_id,
                            req.group_id,
                            req.group_size,
                        );
                    }
                }
                None => break,
            }
        }
        gpu_state.dreqs.append(&mut deferredReq);
    }
    let mut gather_reqs: Vec<(u32, u32, Vec<LDReq>)> = Vec::new();
    let mut sample_reqs: Vec<SampleReq> = Vec::new();
    let mut didSomeWork = false;
    // @Fetch-Submit part
    // @TODO: Refactor the loop - it looks ugly
    {
        // For each compute unit(they run in parallel in our imagination)
        for (cu_id, cu) in &mut gpu_state.cus.iter_mut().enumerate() {
            let mut sampler_free_slots = cu.sampler.get_free_slots();
            // Clear some flags
            for wave in &mut cu.waves {
                wave.has_been_dispatched = false;
                wave.stalled = false;
            }
            // For each fetch-exec unit
            for fe in &mut cu.fes {
                for (wave_id, wave) in &mut cu.waves.iter_mut().enumerate() {
                    if wave.enabled && !wave.has_been_dispatched && !wave.stalled {
                        wave.clock_counter += 1;
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
                                Operand::Sampler(s) => {}
                                Operand::RMemory(mr) => {}
                                Operand::RWMemory(mr) => {}
                                Operand::NONE => {}
                                _ => {
                                    std::panic!("");
                                }
                            }
                        }
                        match &inst.ty {
                            InstTy::ADD
                            | InstTy::SUB
                            | InstTy::MUL
                            | InstTy::DIV
                            | InstTy::LT
                            | InstTy::OR
                            | InstTy::AND
                            | InstTy::ST
                            | InstTy::LD => {
                                inst.assertThreeOp();
                            }
                            InstTy::SAMPLE => {
                                inst.assertFourOp();
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
                            events.push(Event::WAVE_RETIRED((cu_id as u32, wave_id as u32)));
                        } else if let InstTy::LD = inst.ty {
                            let mut wave_gather_reqs: Vec<LDReq> = Vec::new();
                            let addr = wave.getValues(&inst.ops[2]);
                            if let Operand::VRegister(dst) = &inst.ops[0] {
                                for (i, item) in wave.vgprfs[dst.id as usize].iter().enumerate() {
                                    if wave.exec_mask[i] {
                                        let mem_val = match &inst.ops[1] {
                                            Operand::RMemory(rm) => {
                                                match &wave.r_views[rm.id as usize] {
                                                    View::BUFFER(buf) => {
                                                        let mem_offset = buf.offset + addr[i][0];
                                                        let val = applyReadSwizzle(
                                                            &[
                                                                mem_offset,
                                                                mem_offset + 4,
                                                                mem_offset + 8,
                                                                mem_offset + 12,
                                                            ],
                                                            &rm.comps,
                                                        );
                                                        // Boundary checks
                                                        for i in 0..4 {
                                                            if val[i] > 0 {
                                                                assert!(
                                                                    val[i] >= buf.offset
                                                                        && val[i]
                                                                            < buf.offset + buf.size
                                                                );
                                                            }
                                                        }
                                                        val
                                                    }
                                                    _ => std::panic!(""),
                                                }
                                            }
                                            _ => std::panic!(""),
                                        };
                                        let mut address: Value = [0, 0, 0, 0];
                                        applyWriteSwizzle(&mut address, &mem_val, &dst.comps);
                                        for (j, &comp) in address.iter().enumerate() {
                                            if comp != 0 {
                                                // Align at dword
                                                assert!(comp % 4 == 0);
                                                wave_gather_reqs.push(LDReq {
                                                    reg_row: dst.id,
                                                    reg_col: (i * 4 + j) as u32,
                                                    mem_offset: comp,
                                                    timer: gpu_state.config.L1_latency,
                                                });
                                            }
                                        }
                                    }
                                }
                            } else {
                                std::panic!("")
                            }
                            gather_reqs.push((cu_id as u32, wave_id as u32, wave_gather_reqs));
                            wave.pc += 1;
                            wave.has_been_dispatched = true;
                        } else if let InstTy::SAMPLE = inst.ty {
                            // sample dst, res, sampler, coords
                            let sampler_id = match &inst.ops[2] {
                                Operand::Sampler(id) => *id,
                                _ => std::panic!(""),
                            };
                            // let sampler_id = wave.samplers[sampler_id as usize].clone();
                            let (texture_id, read_comps) = match &inst.ops[1] {
                                Operand::RMemory(rm) => match &wave.r_views[rm.id as usize] {
                                    View::TEXTURE2D(tex) => (rm.id, &rm.comps),
                                    _ => std::panic!(""),
                                },
                                _ => std::panic!(""),
                            };
                            let addresses = wave
                                .getValues(&inst.ops[3])
                                .iter()
                                .map(|v| castToFValue(&v))
                                .collect::<Vec<_>>();
                            let mut wave_sample_request: Vec<SampleReq> = Vec::new();
                            if let Operand::VRegister(dst) = &inst.ops[0] {
                                for (i, item) in wave.vgprfs[dst.id as usize].iter_mut().enumerate()
                                {
                                    if wave.exec_mask[i] {
                                        wave_sample_request.push(SampleReq {
                                            cu_id: cu_id as u32,
                                            wave_id: wave_id as u32,
                                            reg_row: dst.id,
                                            reg_col: i as u32,
                                            read_comps: read_comps.clone(),
                                            write_comps: dst.comps.clone(),
                                            u: addresses[i][0],
                                            v: addresses[i][1],
                                            texture: texture_id,
                                            sampler: sampler_id,
                                        });
                                        item.locked = true;
                                    }
                                }
                            } else {
                                std::panic!("")
                            };
                            // Allocate requests in the sampler
                            if sampler_free_slots >= wave_sample_request.len() as u32 {
                                sampler_free_slots =
                                    sampler_free_slots - wave_sample_request.len() as u32;
                                for req in wave_sample_request {
                                    sample_reqs.push(req);
                                }
                                wave.pc += 1;
                                wave.has_been_dispatched = true;
                            }
                        } else if let InstTy::ST = inst.ty {
                            let addr = wave.getValues(&inst.ops[1]);
                            if let Operand::VRegister(src) = &inst.ops[2] {
                                for (i, item) in wave.vgprfs[src.id as usize].iter_mut().enumerate()
                                {
                                    if wave.exec_mask[i] {
                                        // item.locked = true;
                                        let (mem_offsets, mem_vals) = match &inst.ops[0] {
                                            Operand::RWMemory(rm) => {
                                                match &wave.rw_views[rm.id as usize] {
                                                    View::BUFFER(buf) => {
                                                        // Align at dword
                                                        assert!(addr[i][0] % 4 == 0);
                                                        let mem_offset =
                                                            (buf.offset + addr[i][0]) / 4;
                                                        let mut val: Value = [0, 0, 0, 0];
                                                        applyWriteSwizzle(
                                                            &mut val,
                                                            &[
                                                                mem_offset,
                                                                mem_offset + 1,
                                                                mem_offset + 2,
                                                                mem_offset + 3,
                                                            ],
                                                            &rm.comps,
                                                        );
                                                        // Boundary checks
                                                        for i in 0..4 {
                                                            if val[i] > 0 {
                                                                assert!(
                                                                    val[i] * 4 >= buf.offset
                                                                        && val[i] * 4
                                                                            < buf.offset + buf.size
                                                                );
                                                            }
                                                        }
                                                        let regval =
                                                            applyReadSwizzle(&item.val, &src.comps);
                                                        (val, regval)
                                                    }
                                                    View::TEXTURE2D(tex) => {
                                                        let mem_offset =
                                                            (tex.offset + addr[i][0]) / 4;
                                                        let mut mem_offsets: Value = [0, 0, 0, 0];
                                                        let regval =
                                                            applyReadSwizzle(&item.val, &src.comps);
                                                        let mut mem_vals: Value = [0, 0, 0, 0];
                                                        match tex.format {
                                                            TextureFormat::RGBA8_UNORM => {
                                                                mem_offsets[0] = (tex.offset
                                                                    + tex.pitch * addr[i][1]
                                                                    + addr[i][0] * 4)
                                                                    / 4;
                                                                mem_vals[0] = unsafe {
                                                                    (((std::mem::transmute_copy::<u32,f32>(&regval[0]) * 255.0 ) as u32) << 24) |
                                                                    (((std::mem::transmute_copy::<u32,f32>(&regval[1]) * 255.0 ) as u32) << 16) |
                                                                    (((std::mem::transmute_copy::<u32,f32>(&regval[2]) * 255.0 ) as u32) << 8) |
                                                                    (((std::mem::transmute_copy::<u32,f32>(&regval[3]) * 255.0 ) as u32) << 0)
                                                                };
                                                            }
                                                            _ => std::panic!(""),
                                                        };
                                                        (mem_offsets, mem_vals)
                                                    }
                                                    _ => std::panic!(""),
                                                }
                                            }
                                            _ => std::panic!(""),
                                        };

                                        for i in 0..4 {
                                            if mem_offsets[i] != 0 {
                                                gpu_state.mem[mem_offsets[i] as usize] =
                                                    mem_vals[i];
                                            }
                                        }
                                    }
                                }
                            } else {
                                std::panic!("")
                            }
                            wave.pc += 1;
                            wave.has_been_dispatched = true;
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
                                if !valu.ready() {
                                    continue;
                                }
                                wave.has_been_dispatched = true;
                                let mut dispInst = DispInstruction {
                                    exec_mask: Some(wave.exec_mask.clone()),
                                    src: [None, None, None],
                                    instr: Some(inst.clone()),
                                    timer: getLatency(&inst.ty),
                                    wave_id: wave_id as u32,
                                };
                                // Evaluate the source
                                for (i, op) in inst.ops[1..].iter().enumerate() {
                                    match op {
                                        Operand::VRegister(vreg) => {
                                            dispInst.src[i] = Some(wave.getValues(op));
                                        }
                                        Operand::Immediate(imm) => {
                                            dispInst.src[i] = Some(wave.getValues(op));
                                        }
                                        Operand::Builtin(imm) => {
                                            dispInst.src[i] = Some(wave.getValues(op));
                                        }

                                        _ => {}
                                    }
                                }
                                // Lock the destination register
                                match &inst.ty {
                                    InstTy::ADD
                                    | InstTy::SUB
                                    | InstTy::MUL
                                    | InstTy::DIV
                                    | InstTy::OR
                                    | InstTy::AND
                                    | InstTy::LT
                                    | InstTy::MOV
                                    | InstTy::UTOF => {
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
                                assert!(valu.push(&dispInst));
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
                                    events
                                        .push(Event::WAVE_RETIRED((cu_id as u32, wave_id as u32)));

                                    // Wave has been retired
                                }
                            } else {
                                // Not enough resources to dispatch a command
                            }
                        } else if hasSRegOps {
                            std::panic!();
                        } else if dispatchOnSampler {
                            std::panic!();
                        } else {
                            std::panic!();
                        }

                        // Successfully dispatched
                        break;
                    }
                }
            }
            // @ALU
            // Now do work on Vector ALUs
            for valu in &mut cu.valus {
                let inst = valu.pop();
                valu.active = false;
                if inst.is_some() {
                    valu.active = true;
                    didSomeWork = true;

                    let dispInst = inst.unwrap();
                    let mut wave = &mut cu.waves[dispInst.wave_id as usize];
                    let exec_mask = &dispInst.exec_mask.as_ref().unwrap();
                    let inst = dispInst.instr.unwrap();
                    match &inst.ty {
                        InstTy::ADD
                        | InstTy::SUB
                        | InstTy::MUL
                        | InstTy::DIV
                        | InstTy::LT
                        | InstTy::OR
                        | InstTy::AND => {
                            let src1 = dispInst.src[0].as_ref().unwrap();
                            let src2 = dispInst.src[1].as_ref().unwrap();
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
                                            InstTy::OR => ORU32(&x1, &x2),
                                            InstTy::AND => ANDU32(&x1, &x2),
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
                                    applyWriteSwizzle(&mut item.val, &result[i], &dst.comps);
                                    item.locked = false;
                                }
                            }
                        }
                        InstTy::MOV | InstTy::UTOF => {
                            let src1 = dispInst.src[0].as_ref().unwrap();
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
                                    applyWriteSwizzle(&mut item.val, &src, &dst.comps);
                                    item.locked = false;
                                }
                            }
                        }
                        _ => std::panic!("unsupported {:?}", inst.ops[0]),
                    };
                }
            }
            // And on Scalar ALUs
            for salu in &mut cu.salus {}
        }
    }
    // Put memory requests
    for req in sample_reqs {
        assert!(gpu_state.sample(&req));
    }
    for (cu_id, wave_id, gather) in gather_reqs {
        gpu_state.wave_gather(cu_id, wave_id, &gather);
    }
    for cu_id in 0..gpu_state.cus.len() {
        gpu_state.l1_clock(cu_id as u32);
        gpu_state.sampler_clock(cu_id as u32);
    }
    gpu_state.l2_clock();
    gpu_state.mem_clock();
    gpu_state.clock_counter += 1;
    if didSomeWork {
        Some(events)
    } else {
        None
    }
}

#[macro_use]
extern crate lazy_static;
extern crate regex;
use regex::Regex;

fn parse(text: &str) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::new();
    lazy_static! {
        static ref VRegRE: Regex = Regex::new(r"r([0-9]+)\.([xyzw]+)").unwrap();
        static ref RMemRE: Regex = Regex::new(r"t([0-9]+)\.([xyzw]+)").unwrap();
        static ref RWMemRE: Regex = Regex::new(r"u([0-9]+)\.([xyzw]+)").unwrap();
        static ref SamplerRE: Regex = Regex::new(r"s([0-9]+)").unwrap();
        static ref V4FRE: Regex =
            Regex::new(r"f4[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V2FRE: Regex = Regex::new(r"f2[ ]*\([ ]*([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V3FRE: Regex = Regex::new(r"f3[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V1FRE: Regex = Regex::new(r"f[ ]*\([ ]*([^ ]+)[ ]*\)").unwrap();
        static ref V4URE: Regex =
            Regex::new(r"u4[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V2URE: Regex = Regex::new(r"u2[ ]*\([ ]*([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V3URE: Regex = Regex::new(r"u3[ ]*\([ ]*([^ ]+) ([^ ]+) ([^ ]+)[ ]*\)").unwrap();
        static ref V1URE: Regex = Regex::new(r"u[ ]*\([ ]*([^ ]+)[ ]*\)").unwrap();
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
    let parseu32 = |s: &str| -> u32 {
        let without_prefix = s.trim_start_matches("0x");
        if without_prefix != s {
            u32::from_str_radix(without_prefix, 16).unwrap()
        } else {
            without_prefix.parse::<u32>().unwrap()
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
        } else if let Some(x) = RMemRE.captures(s) {
            return Operand::RMemory({
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

                RMemRef {
                    id: regnum.parse::<u32>().unwrap(),
                    comps: [
                        swizzle[0].clone(),
                        swizzle[1].clone(),
                        swizzle[2].clone(),
                        swizzle[3].clone(),
                    ],
                }
            });
        } else if let Some(x) = RWMemRE.captures(s) {
            return Operand::RWMemory({
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

                RWMemRef {
                    id: regnum.parse::<u32>().unwrap(),
                    comps: [
                        swizzle[0].clone(),
                        swizzle[1].clone(),
                        swizzle[2].clone(),
                        swizzle[3].clone(),
                    ],
                }
            });
        } else if let Some(x) = SamplerRE.captures(s) {
            return Operand::Sampler(parseu32(x.get(1).unwrap().as_str()));
        } else if let Some(x) = V4URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V4U(
                    parseu32(x.get(1).unwrap().as_str()),
                    parseu32(x.get(2).unwrap().as_str()),
                    parseu32(x.get(3).unwrap().as_str()),
                    parseu32(x.get(4).unwrap().as_str()),
                )
            });
        } else if let Some(x) = V3URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V3U(
                    parseu32(x.get(1).unwrap().as_str()),
                    parseu32(x.get(2).unwrap().as_str()),
                    parseu32(x.get(3).unwrap().as_str()),
                )
            });
        } else if let Some(x) = V2URE.captures(s) {
            return Operand::Immediate({
                ImmediateVal::V2U(
                    parseu32(x.get(1).unwrap().as_str()),
                    parseu32(x.get(2).unwrap().as_str()),
                )
            });
        } else if let Some(x) = V1URE.captures(s) {
            return Operand::Immediate({ ImmediateVal::V1U(parseu32(x.get(1).unwrap().as_str())) });
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
        let operands: Vec<String> = if command.len() + 1 > line.len() {
            Vec::new()
        } else {
            line[command.len() + 1..]
                .split(",")
                .map(|s| String::from(garbageRE.replace_all(s, "")))
                .filter(|s| s != "")
                .collect::<Vec<String>>()
        };
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
            "ld" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let srcRef = parseOperand(&operands[1]);
                let addrRef = parseOperand(&operands[2]);
                Instruction {
                    ty: match command.as_str() {
                        "ld" => InstTy::LD,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [dstRef, srcRef, addrRef, Operand::NONE],
                }
            }
            "st" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let srcRef = parseOperand(&operands[2]);
                let addrRef = parseOperand(&operands[1]);
                Instruction {
                    ty: match command.as_str() {
                        "st" => InstTy::ST,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [dstRef, addrRef, srcRef, Operand::NONE],
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

            "add.f32" | "sub.f32" | "mul.f32" | "div.f32" | "lt.f32" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let src1Ref = parseOperand(&operands[1]);
                let src2Ref = parseOperand(&operands[2]);
                Instruction {
                    ty: match command.as_str() {
                        "add.f32" => InstTy::ADD,
                        "sub.f32" => InstTy::SUB,
                        "mul.f32" => InstTy::MUL,
                        "div.f32" => InstTy::DIV,
                        "lt.f32" => InstTy::LT,
                        _ => std::panic!(""),
                    },
                    interp: Interpretation::F32,
                    line: line_num as u32,
                    ops: [dstRef, src1Ref, src2Ref, Operand::NONE],
                }
            }
            "add.u32" | "sub.u32" | "mul.u32" | "div.u32" | "lt.u32" | "and" | "or" => {
                assert!(operands.len() == 3);
                let dstRef = parseOperand(&operands[0]);
                let src1Ref = parseOperand(&operands[1]);
                let src2Ref = parseOperand(&operands[2]);
                Instruction {
                    ty: match command.as_str() {
                        "add.u32" => InstTy::ADD,
                        "sub.u32" => InstTy::SUB,
                        "mul.u32" => InstTy::MUL,
                        "div.u32" => InstTy::DIV,
                        "lt.u32" => InstTy::LT,
                        "and" => InstTy::AND,
                        "or" => InstTy::OR,
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
            "sample" => {
                assert!(operands.len() == 4);
                let dstRef = parseOperand(&operands[0]);
                let res = parseOperand(&operands[1]);
                let sampler = parseOperand(&operands[2]);
                let coord = parseOperand(&operands[3]);
                // @TODO: Type check
                Instruction {
                    ty: InstTy::SAMPLE,
                    interp: Interpretation::NONE,
                    line: line_num as u32,
                    ops: [dstRef, res, sampler, coord],
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
        InstTy::ADD
        | InstTy::SUB
        | InstTy::MOV
        | InstTy::UTOF
        | InstTy::LT
        | InstTy::AND
        | InstTy::OR => 1,
        InstTy::MUL => 2,
        InstTy::DIV => 4,
        _ => panic!(""),
    }
}

#[derive(Clone, Debug)]
enum Event {
    GROUP_DISPATCHED((u32, Vec<(u32, u32)>)),
    // (cu_id, wave_id)
    WAVE_STALLED((u32, u32)),
    // (cu_id, wave_id)
    WAVE_RETIRED((u32, u32)),
    // (group)
    GROUP_RETIRED(u32),
    // (cu_id, wave_id, pc)
    INST_DISPATCHED((u32, u32, u32)),
    // (cu_id, wave_id, pc)
    INST_RETIRED((u32, u32)),
}

extern crate image;
use image::{GenericImage, GenericImageView, ImageBuffer, RgbImage};

fn save_image(gpu_state: &GPUState, view: &Texture2D, name: &str) {
    let mut img: RgbImage = ImageBuffer::new(view.width, view.height);
    for i in 0..view.height {
        for j in 0..view.width {
            let raw_val = gpu_state.mem[(view.offset / 4 + view.pitch / 4 * i + j) as usize];
            let comps = [
                (raw_val >> 24) & 0xff,
                (raw_val >> 16) & 0xff,
                (raw_val >> 8) & 0xff,
                (raw_val >> 0) & 0xff,
            ];
            img.put_pixel(
                j,
                i,
                image::Pixel::from_channels(
                    comps[0] as u8,
                    comps[1] as u8,
                    comps[2] as u8,
                    comps[3] as u8,
                ),
            );
        }
    }
    img.save(name).unwrap();
}

#[wasm_bindgen(start)]
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    //alert("PANIC!");
}

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(s: &str) {
    alert(&format!("Hello from guppy_rust, {}!", s));
}

#[derive(Clone, Debug)]
struct BindingState {
    r_views: Vec<View>,
    rw_views: Vec<View>,
    samplers: Vec<Sampler>,
}

static mut g_gpu_state: Option<Box<GPUState>> = None;
static mut g_bind_state: Option<BindingState> = None;

#[wasm_bindgen]
pub fn guppy_create_gpu_state(config_str: String) {
    let config: GPUConfig = serde_json::from_str(config_str.as_str()).unwrap();
    unsafe {
        g_bind_state = Some(BindingState {
            r_views: Vec::new(),
            rw_views: Vec::new(),
            samplers: Vec::new(),
        });
        g_gpu_state = Some(Box::new(GPUState::new(&config)));
        for i in 0..16 {
            g_gpu_state.as_mut().unwrap().mem.push(0);
        }
    }
}

#[wasm_bindgen]
pub fn guppy_get_config() -> String {
    unsafe { serde_json::to_string(&g_gpu_state.as_ref().unwrap().config).unwrap() }
}

#[wasm_bindgen]
pub fn guppy_dispatch(text: &str, group_size: u32, groups_count: u32) {
    // alert("hello1");

    let res = parse(text);;
    let program = Program { ins: res };
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };
    let bind_state = unsafe { g_bind_state.as_mut().unwrap() };
    alert(&format!("bind_state {:?}!", bind_state));
    dispatch(
        gpu_state,
        &program,
        bind_state.r_views.clone(),
        bind_state.rw_views.clone(),
        vec![Sampler {
            wrap_mode: WrapMode::WRAP,
            sample_mode: SampleMode::BILINEAR,
        }],
        group_size,
        groups_count,
    );
}

#[wasm_bindgen]
pub fn guppy_get_gpu_metric(name: String) -> f64 {
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };
    match name.as_str() {
        "ALU active" => gpu_state.get_alu_active(),
        _ => std::panic!(),
    }
}

#[wasm_bindgen]
pub fn guppy_clock() -> bool {
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };
    clock(gpu_state).is_some()
}

extern crate base64;

use base64::{decode, encode};
extern crate png;
use png::{Decoder, Encoder};

fn from_base64(base64: String) -> Vec<u8> {
    let offset = base64.find(',').unwrap_or(base64.len()) + 1;
    let mut value = base64;
    value.drain(..offset);
    return decode(value.as_str()).unwrap();
}

#[wasm_bindgen]
pub fn guppy_init_framebuffer(width: u32, height: u32) -> u32 {
    let bind_state = unsafe { g_bind_state.as_mut().unwrap() };
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };

    bind_state.rw_views.push(View::TEXTURE2D(Texture2D {
        offset: gpu_state.mem.len() as u32 * 4,
        pitch: width * 4,
        width: width,
        height: height,
        format: TextureFormat::RGBA8_UNORM,
    }));
    for i in 0..width * height {
        gpu_state.mem.push(0);
    }

    bind_state.rw_views.len() as u32 - 1
}

#[wasm_bindgen]
pub fn guppy_put_image(base64: String) -> u32 {
    let bytes = std::io::Cursor::new(from_base64(base64));
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().unwrap();
    assert_eq!(info.width as i32 & -(info.width as i32), info.width as i32);
    assert_eq!(
        info.height as i32 & -(info.height as i32),
        info.height as i32
    );
    let mut image = vec![0; info.buffer_size()];
    reader.next_frame(&mut image).unwrap();
    let bind_state = unsafe { g_bind_state.as_mut().unwrap() };
    assert_eq!(image.len() % 4, 0);
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };

    bind_state.r_views.push(View::TEXTURE2D(Texture2D {
        offset: gpu_state.mem.len() as u32 * 4,
        pitch: info.width * 4,
        width: info.width,
        height: info.height,
        format: TextureFormat::RGBA8_UNORM,
    }));
    for i in 0..image.len() / 4 {
        let b0 = image[i * 4];
        let b1 = image[i * 4 + 1];
        let b2 = image[i * 4 + 2];
        let b3 = image[i * 4 + 3];
        let val: u32 = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24);
        gpu_state.mem.push(val);
    }
    // alert(&format!(
    //     "image is created! size: {} x {} first pixel: {:x}!",
    //     info.width, info.height, bind_state.mem[64]
    // ));

    bind_state.r_views.len() as u32 - 1
}

#[wasm_bindgen]
pub fn guppy_get_image(id: u32, read: bool) -> String {
    let bind_state = unsafe { g_bind_state.as_mut().unwrap() };
    let tex2d = if read {
        match &bind_state.r_views[id as usize] {
            View::TEXTURE2D(tex) => tex,
            _ => std::panic!(),
        }
    } else {
        match &bind_state.rw_views[id as usize] {
            View::TEXTURE2D(tex) => tex,
            _ => std::panic!(),
        }
    };
    let mut buf: Vec<u8> = Vec::new();
    let format = match tex2d.format {
        TextureFormat::RGBA8_UNORM => TextureFormat::RGBA8_UNORM,
        _ => std::panic!(),
    };
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };

    for i in 0..tex2d.height {
        for j in 0..tex2d.width {
            let pixel = gpu_state.mem[((tex2d.offset + i * tex2d.pitch + j * 4) / 4) as usize];
            let b0 = ((pixel) & 0xff) as u8;
            let b1 = ((pixel >> 8) & 0xff) as u8;
            let b2 = ((pixel >> 16) & 0xff) as u8;
            let b3 = ((pixel >> 24) & 0xff) as u8;
            buf.push(b0);
            buf.push(b1);
            buf.push(b2);
            buf.push(b3);
        }
    }
    let mut bytes: Vec<u8> = Vec::new();
    let encoder = image::png::PNGEncoder::new(&mut bytes);
    // png::Encoder::new(bytes, tex2d.width,tex2d.height);
    encoder
        .encode(&buf, tex2d.width, tex2d.height, image::ColorType::RGBA(8))
        .expect("error encoding png");
    //encode(&buf, tex2d.width,tex2d.height, png::ColorType::RGBA);
    encode(&bytes)
}

#[wasm_bindgen]
pub fn guppy_get_active_mask() -> Vec<u8> {
    let gpu_state = unsafe { g_gpu_state.as_mut().unwrap() };
    let mut active_mask: Vec<u8> = Vec::new();
    let wave_size = gpu_state.config.wave_size;
    for cu in &gpu_state.cus {
        for wave in &cu.waves {
            if wave.enabled {
                for bit in &wave.exec_mask {
                    if wave.stalled {
                        active_mask.push(3);
                    } else if !wave.has_been_dispatched {
                        active_mask.push(4);
                    } else {
                        if *bit {
                            active_mask.push(1);
                        } else {
                            active_mask.push(0);
                        }
                    }
                }
            } else {
                for i in 0..wave_size {
                    active_mask.push(2);
                }
            }
        }
    }
    active_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_dummy() {
        let res = parse(
            r"
            mov r0.xy, thread_id
            and r0.x, r0.x, u(1023)
            div.u32 r0.y, r0.y, u(1024)
            mov r0.zw, r0.xy
            utof r0.xy, r0.xy
            ; add 0.5 to fit the center of the texel
            add.f32 r0.xy, r0.xy, f2(0.5 0.5)
            ; normalize coordinates
            div.f32 r0.xy, r0.xy, f2(1024.0 1024.0)
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
            ret",
        );
        let config = GPUConfig {
            DRAM_latency: 1,
            DRAM_bandwidth: 2048,
            L1_size: 1 << 14,
            L1_latency: 1,
            L2_size: 1 << 15,
            L2_latency: 1,
            sampler_cache_size: 1 << 10,
            sampler_latency: 1,
            VGPRF_per_pe: 8,
            wave_size: 32,
            CU_count: 12,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        let TEXTURE_SIZE = 64;
        let AMP_K = 16;
        {
            let mut mem: Vec<u32> = Vec::new();
            for i in 0..16 {
                mem.push(0);
            }
            for i in 0..(TEXTURE_SIZE * TEXTURE_SIZE) {
                if i % (TEXTURE_SIZE / 4) == 0 || i % (TEXTURE_SIZE * 4) < TEXTURE_SIZE {
                    mem.push(((i * TEXTURE_SIZE) << 16) | 0xff);
                } else {
                    mem.push(((i * TEXTURE_SIZE) << 8) | 0xff);
                }
            }
            for i in 0..(AMP_K * AMP_K * TEXTURE_SIZE * TEXTURE_SIZE) {
                mem.push(0);
            }
            mem[16] = 0xffffffff;
            mem[17] = 0xff0000ff;
            gpu_state.mem = mem;
        }
        let program = Program { ins: res };
        let out_view = Texture2D {
            offset: 64 + TEXTURE_SIZE * TEXTURE_SIZE * 4,
            pitch: AMP_K * TEXTURE_SIZE * 4,
            width: AMP_K * TEXTURE_SIZE,
            height: AMP_K * TEXTURE_SIZE,
            format: TextureFormat::RGBA8_UNORM,
        };
        let in_view = Texture2D {
            offset: 64,
            pitch: TEXTURE_SIZE * 4,
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            format: TextureFormat::RGBA8_UNORM,
        };
        dispatch(
            &mut gpu_state,
            &program,
            vec![View::TEXTURE2D(in_view.clone())],
            vec![View::TEXTURE2D(out_view.clone())],
            vec![Sampler {
                wrap_mode: WrapMode::WRAP,
                sample_mode: SampleMode::BILINEAR,
            }],
            32,
            (TEXTURE_SIZE * TEXTURE_SIZE * AMP_K * AMP_K) / 32,
        );
        while let Some(events) = clock(&mut gpu_state) {
            if events.len() != 0 {
                // println!("{:?}", events);
                //}

                for event in &events {
                    match event {
                        Event::WAVE_RETIRED((cu_id, wave_id)) => {
                            print!(
                                " clocks:{}",
                                gpu_state.cus[*cu_id as usize].waves[*wave_id as usize]
                                    .clock_counter
                            );
                        }
                        _ => {}
                    }
                }
                println!();
            }
            // println!(
            //                 "{:?},",
            //                 gpu_state.cus[0].sampler.free_reqs.len()
            //                 // gpu_state.cus[0].waves[1].vgprfs[1].iter().map(|reg| {
            //                 //             if reg.locked {
            //                 //                 1
            //                 //             } else {
            //                 //                 0
            //                 //             }
            //                 //         }).collect::<Vec<_>>()
            //             );
            //println!("{:?}", gpu_state.get_alu_active());
        }
        save_image(&gpu_state, &in_view, "in.png");
        save_image(&gpu_state, &out_view, "out.png");
    }

    #[test]
    fn mem_test_texture() {
        let res = parse(
            r"
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
                ",
        );
        let config = GPUConfig {
            DRAM_latency: 4,
            DRAM_bandwidth: 64 * 32,
            L1_size: 1 << 10,
            L1_latency: 4,
            L2_size: 1 << 20,
            L2_latency: 4,
            sampler_cache_size: 1 << 20,
            sampler_latency: 4,
            VGPRF_per_pe: 8,
            wave_size: 32,
            CU_count: 8,
            ALU_per_cu: 4,
            waves_per_cu: 8,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        let w = 0xffffffff as u32;
        let TEXTURE_SIZE = 8;
        let AMP_K = 8;
        {
            let mut mem: Vec<u32> = Vec::new();
            for i in 0..16 {
                mem.push(0);
            }
            for i in 0..(TEXTURE_SIZE * TEXTURE_SIZE) {
                if i % (TEXTURE_SIZE / 4) == 0 || i % (TEXTURE_SIZE * 4) < TEXTURE_SIZE {
                    mem.push(((i * TEXTURE_SIZE) << 16) | 0xff);
                } else {
                    mem.push(((i * TEXTURE_SIZE) << 8) | 0xff);
                }
            }
            for i in 0..(AMP_K * AMP_K * TEXTURE_SIZE * TEXTURE_SIZE) {
                mem.push(0);
            }
            mem[16] = 0xffffffff;
            mem[17] = 0xff0000ff;
            gpu_state.mem = mem;
        }
        let program = Program { ins: res };
        let out_view = Texture2D {
            offset: 64 + TEXTURE_SIZE * TEXTURE_SIZE * 4,
            pitch: AMP_K * TEXTURE_SIZE * 4,
            width: AMP_K * TEXTURE_SIZE,
            height: AMP_K * TEXTURE_SIZE,
            format: TextureFormat::RGBA8_UNORM,
        };
        let in_view = Texture2D {
            offset: 64,
            pitch: TEXTURE_SIZE * 4,
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            format: TextureFormat::RGBA8_UNORM,
        };
        dispatch(
            &mut gpu_state,
            &program,
            vec![View::TEXTURE2D(in_view.clone())],
            vec![View::TEXTURE2D(out_view.clone())],
            vec![Sampler {
                wrap_mode: WrapMode::WRAP,
                sample_mode: SampleMode::BILINEAR,
            }],
            32,
            (TEXTURE_SIZE * TEXTURE_SIZE * AMP_K * AMP_K) / 32,
        );
        while clock(&mut gpu_state).is_some() {
            let wave = &gpu_state.cus[0].waves[0];
            //println!("{:?}", wave.program.as_ref().unwrap().ins[wave.pc as usize]);
        }

        let line = {
            let mut vec: Vec<Value> = Vec::new();
            for j in 0..out_view.width {
                let raw_val =
                    gpu_state.mem[(out_view.offset / 4 + out_view.pitch / 4 * 32 + j) as usize];
                vec.push([
                    (raw_val >> 24) & 0xff,
                    (raw_val >> 16) & 0xff,
                    (raw_val >> 8) & 0xff,
                    (raw_val >> 0) & 0xff,
                ]);
            }
            vec
        };
        assert_eq!(
            line,
            vec![
                [0, 88, 1, 255],
                [0, 63, 16, 255],
                [0, 44, 26, 255],
                [0, 30, 30, 255],
                [0, 22, 28, 255],
                [0, 19, 21, 255],
                [0, 23, 7, 255],
                [0, 28, 12, 255],
                [0, 45, 31, 255],
                [0, 75, 36, 255],
                [0, 119, 28, 255],
                [0, 176, 6, 254],
                [0, 186, 31, 255],
                [0, 146, 70, 255],
                [0, 102, 104, 255],
                [0, 63, 134, 255],
                [0, 27, 160, 255],
                [0, 4, 173, 255],
                [0, 35, 132, 255],
                [0, 61, 96, 255],
                [0, 84, 63, 255],
                [0, 102, 35, 255],
                [0, 117, 11, 255],
                [0, 107, 10, 255],
                [0, 79, 28, 255],
                [0, 59, 38, 255],
                [0, 47, 41, 254],
                [0, 42, 36, 255],
                [0, 44, 24, 255],
                [36, 82, 39, 255],
                [70, 111, 94, 255],
                [86, 136, 126, 255],
                [86, 160, 126, 255],
                [70, 184, 94, 255],
                [38, 195, 41, 255],
                [0, 150, 41, 255],
                [0, 110, 71, 255],
                [0, 74, 98, 255],
                [0, 41, 121, 255],
                [0, 13, 139, 255],
                [0, 14, 128, 255],
                [0, 37, 95, 255],
                [0, 56, 66, 255],
                [0, 72, 40, 255],
                [0, 83, 19, 255],
                [0, 90, 2, 255],
                [0, 69, 13, 255],
                [0, 48, 24, 255],
                [0, 33, 29, 255],
                [0, 23, 29, 255],
                [0, 19, 24, 255],
                [0, 21, 11, 255],
                [0, 26, 6, 255],
                [0, 39, 28, 255],
                [0, 66, 36, 255],
                [0, 107, 31, 255],
                [0, 161, 12, 255],
                [0, 188, 20, 255],
                [0, 158, 61, 255],
                [0, 113, 96, 255],
                [0, 72, 127, 255],
                [0, 36, 154, 255],
                [0, 3, 176, 255],
                [0, 27, 142, 255]
            ]
        );
        // println!("{:?}", line);
        // println!(
        //     "{:?}",
        //     gpu_state.cus[0].waves[0].vgprfs[0]
        //         .iter()
        //         .map(|r| castToFValue(&r.val))
        //         .collect::<Vec<_>>()
        // );
        // println!(
        //     "{:?}",
        //     gpu_state.cus[0].waves[0].vgprfs[1]
        //         .iter()
        //         .map(|r| castToFValue(&r.val))
        //         .collect::<Vec<_>>()
        // );
    }

    #[test]
    fn mem_test_3() {
        let res = parse(
            r"
                mov r1.w, thread_id
                mul.u32 r1.x, r1.w, u(12)
                ld r2.x, t0.x, r1.x
                add.u32 r1.y, r1.x, u(64)
                and r1.y, r1.y, u(0xff)
                ld r3.x, t0.x, r1.y
                add.u32 r2.x, r2.x, r3.x
                mul.u32 r1.x, r1.w, u(4)
                st u0.x, r1.x, r2.x
                ret
                ",
        );
        let config = GPUConfig {
            DRAM_latency: 4,
            DRAM_bandwidth: 64,
            L1_size: 64,
            L1_latency: 4,
            L2_size: 64,
            L2_latency: 4,
            sampler_cache_size: 1 << 10,
            sampler_latency: 100,
            VGPRF_per_pe: 8,
            wave_size: 16,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 8,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        gpu_state.mem = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // padding
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // read buffer
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // read buffer
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // read buffer
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // read buffer
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // write buffer
        ];
        let program = Program { ins: res };
        dispatch(
            &mut gpu_state,
            &program,
            vec![View::BUFFER(Buffer {
                offset: 64,
                size: 64 * 4,
            })],
            vec![View::BUFFER(Buffer {
                offset: 64 * 5,
                size: 16 * 4,
            })],
            vec![],
            16,
            1,
        );
        let mut history: Vec<String> = Vec::new();
        while clock(&mut gpu_state).is_some() {
            // "\"{:?}:<{:?}|{:?}|{:?}>\",",

            let line = format!(
                "{:?}:<{:?}|{:?}|{:?}>",
                gpu_state.cus[0].waves[0].pc,
                gpu_state.ld_reqs.len(),
                gpu_state.l2.req_queue.len(),
                gpu_state.cus[0].l1.gather_queue.len()
            );
            history.push(line);
        }
        assert_eq!(
            history,
            vec![
                "1:<0|0|0>",
                "2:<0|0|0>",
                "2:<0|0|0>",
                "3:<0|0|3>",
                "4:<0|0|3>",
                "5:<0|0|3>",
                "6:<0|3|4>",
                "6:<0|3|4>",
                "6:<0|3|4>",
                "6:<3|4|4>",
                "6:<3|4|4>",
                "6:<3|4|4>",
                "6:<3|1|1>",
                "6:<2|1|1>",
                "6:<1|1|1>",
                "6:<0|0|0>",
                "7:<0|0|0>",
                "8:<0|0|0>",
                "8:<0|0|0>",
                "9:<0|0|0>",
                "9:<0|0|0>",
            ]
        );
    }

    #[test]
    fn mem_test_2() {
        let res = parse(
            r"
                mov r1.x, thread_id
                mul.u32 r1.x, r1.x, u(4)
                ld r2.x, t0.x, r1.x
                add.u32 r1.y, r1.x, u(128)
                and r1.y, r1.y, u(0xfff)
                ld r6.w, t0.x, r1.y
                add.u32 r2.z, r2.x, r6.w
                st u0.x, r1.x, r2.z
                ret
                ",
        );
        let config = GPUConfig {
            DRAM_latency: 30,
            DRAM_bandwidth: 512,
            L1_size: 1 << 14,
            L1_latency: 10,
            L2_size: 1 << 15,
            L2_latency: 20,
            sampler_cache_size: 1 << 10,
            sampler_latency: 100,
            VGPRF_per_pe: 8,
            wave_size: 32,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        let ITEMS = 1024;
        {
            let mut mem: Vec<u32> = Vec::new();
            mem.append(&mut vec![0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
            for i in 0..ITEMS {
                mem.push(i);
            }
            for i in 0..ITEMS {
                mem.push(0);
            }
            gpu_state.mem = mem;
        }
        let program = Program { ins: res };
        dispatch(
            &mut gpu_state,
            &program,
            vec![View::BUFFER(Buffer {
                offset: 16,
                size: ITEMS * 4,
            })],
            vec![View::BUFFER(Buffer {
                offset: 16 + ITEMS * 4,
                size: ITEMS * 4,
            })],
            vec![],
            128,
            ITEMS / 128,
        );
        while clock(&mut gpu_state).is_some() {
            // println!(
            //     "{:?}:<{:?}|{:?}|{:?}>",
            //     gpu_state.cus[0].waves[0].pc,
            //     gpu_state.ld_reqs.len(),
            //     gpu_state.l2.req_queue.len(),
            //     gpu_state.cus[0].l1.gather_queue.len()
            // );
        }
        for i in 0..ITEMS {
            assert_eq!(
                gpu_state.mem[(4 + ITEMS + i) as usize],
                i + ((i + 32) & (ITEMS - 1))
            );
        }
    }

    #[test]
    fn mem_test() {
        let res = parse(
            r"
                mov r1.x, thread_id
                mul.u32 r1.x, r1.x, u(4)
                ld r2.xy, t0.xw, r1.x
                add.u32 r1.y, r1.x, u(4)
                ld r3.x, t0.x, r1.y
                add.u32 r2.x, r2.x, r2.y
                ;mov r2.x, r2.x
                st u0.x, r1.x, r2.x

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
            VGPRF_per_pe: 8,
            wave_size: 16,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
            ALU_pipe_len: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        gpu_state.mem = vec![
            0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, // padding
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, // read buffer
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // write buffer
        ];
        let program = Program { ins: res };
        dispatch(
            &mut gpu_state,
            &program,
            vec![View::BUFFER(Buffer {
                offset: 16,
                size: 20 * 4,
            })],
            vec![View::BUFFER(Buffer {
                offset: 16 + 20 * 4,
                size: 16 * 4,
            })],
            vec![],
            16,
            1,
        );
        while clock(&mut gpu_state).is_some() {}
        assert_eq!(
            vec![3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 13, 14, 15],
            gpu_state.cus[0].waves[0].vgprfs[2]
                .iter()
                .map(|r| r.val[0])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0],
            gpu_state.cus[0].waves[0].vgprfs[3]
                .iter()
                .map(|r| r.val[0])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 13, 14, 15],
            gpu_state.mem[24..40]
        );
    }

    #[test]
    fn hazard_test() {
        let res = parse(
            r"
                mov r1.x, thread_id
                mov r2.x, r1.x
                utof r2.x, r2.x
                ; RAW hazard
                div.f32 r3.x, r2.x, f(2.0)
                mov r4.x, r3.x
                ; WAR hazard
                div.f32 r5.x, r2.x, f(4.0)
                mov r2.x, f(0.0)

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
            VGPRF_per_pe: 8,
            wave_size: 32,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
            ALU_pipe_len: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program { ins: res };
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 32, 1);
        while clock(&mut gpu_state).is_some() {}
        assert_eq!(
            vec![
                0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                15.0, 15.5
            ],
            gpu_state.cus[0].waves[0].vgprfs[4]
                .iter()
                .map(|r| castToFValue(&r.val)[0])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5,
                3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25,
                7.5, 7.75
            ],
            gpu_state.cus[0].waves[0].vgprfs[5]
                .iter()
                .map(|r| castToFValue(&r.val)[0])
                .collect::<Vec<_>>()
        );
    }

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
            VGPRF_per_pe: 8,
            wave_size: 32,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
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
                ",
            ),
        };
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 16, 1);
        let mut mask_history: Vec<String> = Vec::new();
        while clock(&mut gpu_state).is_some() {
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
            VGPRF_per_pe: 8,
            wave_size: 8,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
                mov r4.w, lane_id
                utof r4.xyzw, r4.wwww
                mov r4.z, wave_id
                utof r4.z, r4.z
                add.f32 r4.xyzw, r4.xyzw, f4(0.0 0.0 0.0 1.0)
                lt.f32 r4.xy, r4.ww, f2(4.0 2.0)
                utof r4.xy, r4.xy
                br_push r4.x, LB_1, LB_2
                mov r0.x, f(666.0)
                br_push r4.y, LB_0_1, LB_0_2
                mov r0.y, f(666.0)
                pop_mask
            LB_0_1:
                mov r0.y, f(777.0)
                pop_mask
            LB_0_2:
                pop_mask
            LB_1:
                mov r0.x, f(777.0)

                ; push the current wave mask
                push_mask LOOP_END
            LOOP_PROLOG:
                lt.f32 r4.x, r4.w, f(8.0)
                add.f32 r4.w, r4.w, f(1.0)
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
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 8, 1);
        let mut mask_history: Vec<Vec<u32>> = Vec::new();
        while clock(&mut gpu_state).is_some() {
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
            VGPRF_per_pe: 8,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 4,
            fd_per_cu: 4,
            ALU_pipe_len: 1,
        };
        // @TODO: 2 cycles are wasted
        let mut gpu_state = GPUState::new(&config);
        let program = Program {
            ins: parse(
                r"
                mov r4.x, f(8.0)
                mov r5.y, f(2.0)
                mov r0.x, f(1.0)
                mov r1.x, f(2.0)
                div.f32 r4.x, r5.y, r4.x
                div.f32 r2.x, r0.x, r1.x
                ret
                ",
            ),
        };
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 4, 1);
        while clock(&mut gpu_state).is_some() {
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
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 8, 1);
        while clock(&mut gpu_state).is_some() {}
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
                add.f32 r4.xyzw, r4.xyzw, f4(1.0 1.0 0.0 1.0)
                lt.f32 r4.xy, r4.ww, f2(3.0 2.0)
                utof r4.xy, r4.xy
                br_push r4.x, LB_1, LB_2
                LB_0:
                mov r0.x, f(666.0)
                br_push r4.y, LB_0_1, LB_0_2
                LB_0_0:
                mov r0.y, f(666.0)
                pop_mask
                LB_0_1:
                mov r0.y, f(777.0)
                pop_mask
                LB_0_2:
                pop_mask
                LB_1:
                mov r0.x, f(777.0)
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
            VGPRF_per_pe: 8,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
            ALU_pipe_len: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let program = Program { ins: res };
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 8, 1);
        while clock(&mut gpu_state).is_some() {}
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
                add.f32 r1.xyzw, r2.xyzw, r3.wzxy
                mov r4.xyzw, f4 ( 1.0 2.0 3.0 5.0 )
                pop_mask
                ret
                LB_1:
                br_push r1.x, LB_2, LB_3
                LB_2:
                pop_mask
                LB_3:
                pop_mask
                ret
                lt.f32 r1.x, r2.x, r3.y
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
            VGPRF_per_pe: 8,
            wave_size: 4,
            CU_count: 2,
            ALU_per_cu: 2,
            waves_per_cu: 16,
            fd_per_cu: 4,
            ALU_pipe_len: 4,
        };
        let mut gpu_state = GPUState::new(&config);
        let mut program = Program { ins: Vec::new() };
        // 1: mov r1.wzyx, f4(1.0f, 1.0f, 1.0f, 0.0f)
        // 2: mov r2.wzyx, f4(1.0f, 2.0f, 3.0f, 4.0f)
        // 3: add_f32 r0.xyzw, r1.xyzw, r2.xyzw
        // 4: mov r1.w, thread_id
        // 5: utof r1.xyzw, r1.wwww
        // 6: add_f32 r3.xyzw, r1.xyzw, r2.xyzw
        // 7: mov r4.w, f(777.0f)
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

        // 8: lt_f32 r1.x, r1.y, f(2.0f)
        // 9: br_push r1.x, LB_1, LB_3
        //10: LB_0:
        //11: mov r1.x, f(1.0f)
        //12: pop
        //13: LB_1:
        //14: mov r1.x, f(0.0f)
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
        dispatch(&mut gpu_state, &program, vec![], vec![], vec![], 4, 1);
        while clock(&mut gpu_state).is_some() {}
    }
}
