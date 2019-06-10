import GoldenLayout from 'golden-layout';
import React from 'react';
import ReactDOM from 'react-dom';
import './css/main.css';
import AceEditor from 'react-ace';
import 'brace/mode/assembly_x86';
// Import a Theme (okadia, github, xcode etc)
import 'brace/theme/tomorrow_night_eighties';
import { JSONEditor } from 'react-json-editor-viewer';

function onChange(newValue) {
    console.log('change', newValue);
}

const _wasm = import("guppy_rust");

class App extends React.Component {

    constructor(props, context) {
        super(props, context);

        this.onChange = this.onChange.bind(this);
        this.Execute = this.Execute.bind(this);
        this.PauseResume = this.PauseResume.bind(this);
        this.onResize = this.onResize.bind(this);
    }

    componentDidMount() {
        // this.refs.editor.setValue(
        //     "ret"
        // );
        this.props.glContainer.on('resize', this.onResize);
    }

    onChange(newValue) {
        this.text = newValue;
    }

    onResize() {
        // this.refs.editor.resize();
    }

    PauseResume() {
        if (this.props.globals().wasm) {
            this.props.globals().run = !this.props.globals().run;
        }
    }

    Execute() {
        let globals = this.props.globals();
        if (this.props.globals().wasm) {
            this.props.globals().resetGPU();
            let config = this.props.globals().dispatchConfig;
            for (var i = 0; i < globals.r_images.length; i++) {
                globals.wasm.guppy_put_image(globals.r_images[i]);
            }
            globals.wasm.guppy_init_framebuffer(256, 256);
            this.props.globals().wasm.guppy_dispatch(
                this.text,
                config["group_size"],
                config["groups_count"]
            );

            this.props.globals().run = true;
        } else {
            console.log("[WARNING] wasm in null");
        }
    }
    render() {
        let def_value =
            "\n\
mov r0.xy, thread_id\n\
and r0.x, r0.x, u(255)\n\
div.u32 r0.y, r0.y, u(256)\n\
mov r0.zw, r0.xy\n\
utof r0.xy, r0.xy\n\
; add 0.5 to fit the center of the texel\n\
add.f32 r0.xy, r0.xy, f2(0.5 0.5)\n\
; normalize coordinates\n\
div.f32 r0.xy, r0.xy, f2(256.0 256.0)\n\
; tx * 2.0 - 1.0\n\
mul.f32 r0.xy, r0.xy, f2(2.0 2.0)\n\
sub.f32 r0.xy, r0.xy, f2(1.0 1.0)\n\
; rotate with pi/4\n\
mul.f32 r4.xy, r0.xy, f2(0.7071 0.7071)\n\
add.f32 r5.x, r4.x, r4.y\n\
sub.f32 r5.y, r4.y, r4.x\n\
;mul.f32 r5.x, r5.x, f(2.0)\n\
mov r0.xy, r5.xy\n\
; texture fetch\n\
; coordinates are normalized\n\
; type conversion and coordinate conversion happens here\n\
sample r1.xyzw, t0.xyzw, s0, r0.xy\n\
; coordinates are u32 here(in texels)\n\
; type conversion needs to happen\n\
st u0.xyzw, r0.zw, r1.xyzw\n\
ret";

        let def_value_0 =
            "\
mov r0.xy, thread_id\n\
and r0.x, r0.x, u(63)\n\
div.u32 r0.y, r0.y, u(64)\n\
mov r0.zw, r0.xy\n\
mov r1.xyzw, f4(1.0 0.0 0.0 1.0)\n\
st u0.xyzw, r0.zw, r1.xyzw\n\
ret\n\
        ";
        let def_value_1 =
            "\
    mov r4.w, lane_id\n\
    utof r4.xyzw, r4.wwww\n\
    mov r4.z, wave_id\n\
    utof r4.z, r4.z\n\
    add.f32 r4.xyzw, r4.xyzw, f4(0.0 0.0 0.0 1.0)\n\
    lt.f32 r4.xy, r4.ww, f2(16.0 8.0)\n\
    utof r4.xy, r4.xy\n\
    br_push r4.x, LB_1, LB_2\n\
    mov r0.x, f(666.0)\n\
    br_push r4.y, LB_0_1, LB_0_2\n\
    mov r0.y, f(666.0)\n\
    pop_mask\n\
LB_0_1:\n\
    mov r0.y, f(777.0)\n\
    pop_mask\n\
LB_0_2:\n\
    pop_mask\n\
LB_1:\n\
    mov r0.x, f(777.0)\n\
    ; push the current wave mask\n\
    push_mask LOOP_END\n\
LOOP_PROLOG:\n\
    lt.f32 r4.x, r4.w, f(24.0)\n\
    add.f32 r4.w, r4.w, f(1.0)\n\
    ; Setting current lane mask\n\
    ; If all lanes are disabled pop_mask is invoked\n\
    ; If mask stack is empty then wave is retired\n\
    mask_nz r4.x\n\
LOOP_BEGIN:\n\
    jmp LOOP_PROLOG\n\
LOOP_END:\n\
    pop_mask\n\
LB_2:\n\
    mov r4.y, lane_id\n\
    utof r4.y, r4.y\n\
    ret";
        this.text = def_value;
        return (
            <div className="ace_editor_container">
                <button style={{ margin: 10 }} onClick={this.Execute}>
                    Execute
                </button>
                <button style={{ margin: 10 }} onClick={this.PauseResume}>
                    Pause/Resume
                </button>
                <AceEditor
                    value={def_value}
                    ref="editor"
                    mode="assembly_x86"
                    theme="tomorrow_night_eighties"
                    onChange={this.onChange}
                    name="UNIQUE_ID_OF_DIV"
                    editorProps={{
                        $blockScrolling: true
                    }}
                    autoScrollEditorIntoView={false}
                    wrapEnabled={false}
                    height="700px"
                    width="512px"
                />
            </div>
        );
    }
}

class Parameters extends React.Component {

    constructor(props, context) {
        super(props, context);

        this.onChange = this.onChange.bind(this);
        this.onChangeDispatch = this.onChangeDispatch.bind(this);
    }

    componentDidMount() {
    }

    onChange(key, value, parent, data) {
        console.log("onchange", key, value);
        this.props.globals().gpuConfig[key] = value;


    }

    onChangeDispatch(key, value, parent, data) {
        console.log("onChangeDispatch", key, value);
        this.props.globals().dispatchConfig[key] = value;

    }

    render() {

        return (
            <div>
                <p style={{ color: "white", margin: 10 }}>GPU Config</p>
                <JSONEditor
                    data={
                        this.props.globals().gpuConfig
                    }
                    onChange={this.onChange}
                />
                <p style={{ color: "white", margin: 10 }}>Dispatch config</p>
                <JSONEditor
                    data={
                        this.props.globals().dispatchConfig
                    }
                    onChange={this.onChangeDispatch}
                />
            </div>
        );
    }
}

class Memory extends React.Component {

    constructor(props, context) {
        super(props, context);

        this.updateMemory = this.updateMemory.bind(this);
    }

    componentDidMount() {
        this.props.globals().updateMemory = this.updateMemory;
        this.ctx = this.refs.canvas.getContext('2d');
        let img = this.refs.img;
        img.onload = () => {
            this.refs.canvas.width = img.width;
            this.refs.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);
            var p = this.ctx.getImageData(0, 0, 1, 1).data;
            function rgbToHex(r, g, b) {
                if (r > 255 || g > 255 || b > 255)
                    throw "Invalid color component";
                return ((r << 16) | (g << 8) | b).toString(16);
            }
            var hex = "#" + ("000000" + rgbToHex(p[0], p[1], p[2])).slice(-6);
            let base64 = this.refs.canvas.toDataURL('image/png');
            this.props.globals().r_images.push(base64);
            // console.log(base64);
        };

    }

    updateMemory() {
        let ctx = this.refs.canvas.getContext('2d');
        let image = new Image();
        let canvas = this.refs.canvas;
        let globals = this.props.globals();
        image.onload = function () {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
        };
        let base64 = "data:image/png;base64," + globals.wasm.guppy_get_image(0, false);
        // console.log(base64);
        image.src = base64;
    }

    render() {

        return (
            <div>
                <canvas ref="canvas" />
                <img style={{ display: "none" }} ref="img" src="img/lenna.png"></img>
            </div>
        );
    }
}

class CanvasComponent extends React.Component {

    constructor(props, context) {
        super(props, context);
        this.neededWidth = 512;
        this.neededHeight = 512;
        this.updateCanvas = this.updateCanvas.bind(this);
        this.onResize = this.onResize.bind(this);
        this.scheduleDraw = this.scheduleDraw.bind(this);
    }

    componentDidMount() {
        this.draw = true;
        this.ctx = this.refs.canvas.getContext('2d');
        this.canvas = this.refs.canvas;
        this.lastClock = 0;
        this.props.glContainer.on('resize', this.onResize);
        this.globals = this.props.globals;
        this.globals().updateCanvas = this.updateCanvas;
        
        this.updateCanvas();
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.updateCanvas)
    }

    onResize() {
        this.updateCanvas();
    }

    scheduleDraw() {
        this.draw = true;
        this.updateCanvas();
    }
    updateCanvas() {
        if (!this.draw) {
            return;
        }
        this.draw = false;
        this.canvas.width = this.neededWidth;
        this.canvas.height = this.neededHeight;
        this.ctx.fillStyle = "#222222";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        var x = 0;
        var y = 0;
        if (this.globals().active_mask_history) {
            let history = this.globals().active_mask_history;
            if (history.length > 0) {
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "exec mask history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, 0, y);
                // console.log('updateCanvas', history[0].length);

                var exec_mask_offset = y + 16;
                for (var i = 0; i < history.length; i++) {
                    this.neededHeight = Math.max(this.neededHeight, y);
                    y = exec_mask_offset;
                    for (var j = 0; j < history[0].length; j++) {
                        if (j % 32 == 0)
                            y += 4;
                        if (history[i][j] == 1) {
                            this.ctx.fillStyle = "white";
                        } else if (history[i][j] == 0) {
                            this.ctx.fillStyle = "black";
                        } else if (history[i][j] == 2) {
                            this.ctx.fillStyle = "grey";
                        } else if (history[i][j] == 3) {
                            this.ctx.fillStyle = "blue";
                        } else if (history[i][j] == 4) {
                            this.ctx.fillStyle = "red";
                        }
                        this.ctx.fillRect(x, y, 1, 1);
                        y += 1;
                    }
                    x += 1;
                }
                this.neededWidth = Math.max(this.neededWidth, x);
            }
        }
        if (this.globals().alu_active_history) {
            let history = this.globals().alu_active_history;
            if (history.length > 0) {
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "ALU Active history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, 0, y + 16);
                var exec_mask_offset = y + 32;
                x = 0;
                for (var i = 0; i < history.length; i++) {
                    this.neededHeight = Math.max(this.neededHeight, y);
                    y = exec_mask_offset;
                    for (var j = 100; j >= 0; j--) {
                        if (j <= history[i]) {
                            this.ctx.fillStyle = "white";
                        } else {
                            this.ctx.fillStyle = "black";
                        }
                        this.ctx.fillRect(x, y, 1, 1);
                        y += 1;
                    }
                    x += 1;
                }
                this.neededWidth = Math.max(this.neededWidth, x);
            }
        }
        if (this.globals().l2_metric_history) {
            let history = this.globals().l2_metric_history;
            if (history.length > 0) {
                
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "L2 metric history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, 0, y + 16);
                var exec_mask_offset = y + 32;
                x = 0;
                var max = 0;
                for (var i = 0; i < history.length; i++) {
                    max = Math.max(max, history[i][0], history[i][1], history[i][2]);
                }
                // console.log(max);
                for (var i = 0; i < history.length; i++) {
                    this.neededHeight = Math.max(this.neededHeight, y);
                    y = exec_mask_offset;
                    let hit = 100.0 * history[i][0] / max;
                    let miss = 100.0 * history[i][1] / max;
                    let evict = 100.0 * history[i][2] / max;
                    for (var j = 100; j >= 0; j--) {
                        var r = 0;
                        var g = 0;
                        var b = 0;
                        if (j <= hit) {
                            r = 255.0;
                        }
                        if (j <= miss) {
                            g = 255.0;
                        }
                        if (j <= evict) {
                            b = 255.0;
                        }
                        this.ctx.fillStyle = 'rgb(' +
                        Math.floor(r) + ', ' +
                        Math.floor(g) + ', ' +
                        Math.floor(b) + ')';
                        this.ctx.fillRect(x, y, 1, 1);
                        y += 1;
                    }
                    x += 1;
                }
                this.neededWidth = Math.max(this.neededWidth, x);
            }
        }
        if (this.globals().samplers_metric_history) {
            let history = this.globals().samplers_metric_history;
            if (history.length > 0) {
                
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "Samplers metric history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, 0, y + 16);
                var exec_mask_offset = y + 32;
                x = 0;
                var max = 0;
                for (var i = 0; i < history.length; i++) {
                    max = Math.max(max, history[i][0], history[i][1], history[i][2]);
                }
                // console.log(max);
                for (var i = 0; i < history.length; i++) {
                    this.neededHeight = Math.max(this.neededHeight, y);
                    y = exec_mask_offset;
                    let hit = 100.0 * history[i][0] / max;
                    let miss = 100.0 * history[i][1] / max;
                    let evict = 100.0 * history[i][2] / max;
                    for (var j = 100; j >= 0; j--) {
                        var r = 0;
                        var g = 0;
                        var b = 0;
                        if (j <= hit) {
                            r = 255.0;
                        }
                        if (j <= miss) {
                            g = 255.0;
                        }
                        if (j <= evict) {
                            b = 255.0;
                        }
                        this.ctx.fillStyle = 'rgb(' +
                        Math.floor(r) + ', ' +
                        Math.floor(g) + ', ' +
                        Math.floor(b) + ')';
                        this.ctx.fillRect(x, y, 1, 1);
                        y += 1;
                    }
                    x += 1;
                }
                this.neededWidth = Math.max(this.neededWidth, x);
            }
        }
    }
    render() {
        return <div>
            <button style={{ margin: 10 }} onClick={this.scheduleDraw}>
                    Draw
            </button>
            <canvas ref="canvas" /> </div>;
    }
}
class GoldenLayoutWrapper extends React.Component {
    timer() {
        if (this.globals.wasm && this.globals.run) {
            for (var i = 0; i < this.globals.dispatchConfig["cycles_per_iter"]; i++) {
                if (!this.globals.wasm.guppy_clock()) {
                    this.globals.run = false;
                } else {
                    this.globals.active_mask_history.push(this.globals.wasm.guppy_get_active_mask());
                    this.globals.alu_active_history.push(this.globals.wasm.guppy_get_gpu_metric("ALU active"));
                    this.globals.samplers_metric_history.push([
                        this.globals.wasm.guppy_get_gpu_metric("Samplers cache hit"),
                        this.globals.wasm.guppy_get_gpu_metric("Samplers cache miss"),
                        this.globals.wasm.guppy_get_gpu_metric("Samplers cache evict"),
                    ]);
                    this.globals.l2_metric_history.push([
                        this.globals.wasm.guppy_get_gpu_metric("L2 hit"),
                        this.globals.wasm.guppy_get_gpu_metric("L2 miss"),
                        this.globals.wasm.guppy_get_gpu_metric("L2 evict"),
                    ]);
                    //if (this.globals.updateCanvas)
                    this.globals.updateMemory();
                    // this.globals.updateCanvas();
                }
            }
        }
    }
    componentDidMount() {
        this.globals = {};
        this.globals.dispatchConfig = {
            "group_size": 32, "groups_count": 2048, "cycles_per_iter": 4 };
        this.globals.gpuConfig = {
            "DRAM_latency": 16,
            "DRAM_bandwidth": 2048, "L1_size": 1024, "L1_latency": 4,
            "L2_size": 8 * 1024, "L2_latency": 8, "sampler_cache_size": 4 * 1024,
            "sampler_latency": 4, "VGPRF_per_pe": 8, "wave_size": 32,
            "CU_count": 4, "ALU_per_cu": 2, "waves_per_cu": 4, "fd_per_cu": 2,
            "ALU_pipe_len": 2
        };
        this.globals.wasm = null;
        this.globals.r_images = [];
        this.globals.active_mask_history = null;
        this.globals.samplers_metric_history = null;
        this.globals.l2_metric_history = null;
        this.globals.alu_active_history = null;

        this.intervalId = setInterval(this.timer.bind(this), 1);
        // Build basic golden-layout config
        const config = {
            content: [{
                type: 'row',
                content: [{
                    type: 'react-component',
                    component: 'TextEditor',
                    title: 'TextEditor',
                    props: { globals: () => this.globals }

                }, {
                    type: 'react-component',
                    component: 'Canvas',
                    title: 'Canvas',
                    props: { globals: () => this.globals }

                },
                {
                    type: 'column',
                    content: [
                        {
                            type: 'react-component',
                            component: 'Parameters',
                            title: 'Parameters',
                            props: { globals: () => this.globals }
                        },
                        {
                            type: 'react-component',
                            component: 'Memory',
                            title: 'Memory',
                            props: { globals: () => this.globals }
                        }
                    ]
                }
                ]
            }]
        };

        var layout = new GoldenLayout(config, this.layout);
        this.layout = layout;
        let globals = this.globals;
        this.globals.resetGPU = function () {
            globals.run = false;
            globals.wasm.guppy_create_gpu_state(
                JSON.stringify(globals.gpuConfig));
            globals.active_mask_history = [];
            globals.alu_active_history = [];
            globals.samplers_metric_history = [];
            globals.l2_metric_history = [];
        }
        _wasm.then(wasm => {
            layout.updateSize();
            this.globals.wasm = wasm;
            this.globals.resetGPU();
            //console.log("wasm loaded");
            //console.log(wasm.guppy_get_config());

        });
        layout.registerComponent('Canvas', CanvasComponent
        );
        layout.registerComponent('TextEditor',
            App
        );
        layout.registerComponent('Parameters',
            Parameters
        );
        layout.registerComponent('Memory',
            Memory
        );
        layout.init();
        window.React = React;
        window.ReactDOM = ReactDOM;
        window.addEventListener('resize', () => {
            layout.updateSize();
        });


        //layout.updateSize();
    }

    render() {
        return (
            <div className='goldenLayout'
                ref={input => this.layout = input} />
        );
    }
}


export default GoldenLayoutWrapper;