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
        this.onClick = this.onClick.bind(this);
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

    onClick() {
        let globals = this.props.globals();
        if (this.props.globals().wasm) {
            this.props.globals().active_mask_history = [];
            this.props.globals().alu_active_history = [];
            let config = this.props.globals().dispatchConfig;
            this.props.globals().wasm.guppy_dispatch(
                this.text,
                config["group_size"],
                config["groups_count"]
            );
            for (var i = 0; i < globals.r_images.length; i++) {
                globals.wasm.guppy_put_image(globals.r_images[i]);
            }
            let ctx = this.refs.canvas.getContext('2d');
            let image = new Image();
            let canvas = this.refs.canvas;
            image.onload = function () {
                canvas.width = image.width;
                canvas.height = image.height;
                ctx.drawImage(image, 0, 0);
            };
            let base64 = "data:image/png;base64," + globals.wasm.guppy_get_image(0);
            console.log(base64);
            image.src = base64;
            this.props.globals().run = true;
        } else {
            console.log("[WARNING] wasm in null");
        }
    }
    render() {
        let def_value =
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
                <button style={{ margin: 10 }} onClick={this.onClick}>
                    Execute
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
                <canvas style={{ display: "none" }} ref="canvas"></canvas>
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
        this.props.globals().resetGPU();

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

        this.onChange = this.onChange.bind(this);
    }

    componentDidMount() {
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
            console.log(base64);
        };

    }

    onChange() {

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
    componentDidMount() {
        this.neededWidth = 512;
        this.neededHeight = 512;
        this.updateCanvas = this.updateCanvas.bind(this);
        this.onResize = this.onResize.bind(this);
        this.ctx = this.refs.canvas.getContext('2d');
        this.canvas = this.refs.canvas;
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

    updateCanvas() {
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
                        this.ctx.fillRect(x, y, 2, 2);
                        y += 2;
                    }
                    x += 2;
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
                        this.ctx.fillRect(x, y, 2, 2);
                        y += 2;
                    }
                    x += 2;
                }
                this.neededWidth = Math.max(this.neededWidth, x);
            }
        }

    }
    render() {
        return <canvas ref="canvas" />;
    }
}

class GoldenLayoutWrapper extends React.Component {
    timer() {
        if (this.globals.wasm && this.globals.run) {
            if (!this.globals.wasm.guppy_clock()) {
                this.globals.run = false;
            } else {
                this.globals.active_mask_history.push(this.globals.wasm.guppy_get_active_mask());
                this.globals.alu_active_history.push(this.globals.wasm.guppy_get_gpu_metric("ALU active"));
                //if (this.globals.updateCanvas)
                this.globals.updateCanvas();
            }
        }
    }
    componentDidMount() {
        this.globals = {};
        this.globals.dispatchConfig = { "group_size": 32, "groups_count": 2 };
        this.globals.gpuConfig = {
            "DRAM_latency": 4,
            "DRAM_bandwidth": 2048, "L1_size": 1024, "L1_latency": 4, "L2_size": 16384, "L2_latency": 4, "sampler_cache_size": 1024,
            "sampler_latency": 4, "VGPRF_per_pe": 8, "wave_size": 32, "CU_count": 2, "ALU_per_cu": 4, "waves_per_cu": 4, "fd_per_cu": 2, "ALU_pipe_len": 1
        };
        this.globals.wasm = null;
        this.globals.r_images = [];
        this.globals.active_mask_history = null;
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
            globals.wasm.guppy_create_gpu_state(
                JSON.stringify(globals.gpuConfig));
            globals.active_mask_history = [];
            globals.alu_active_history = [];
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