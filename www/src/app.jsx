import GoldenLayout from 'golden-layout';
import React from 'react';
import ReactDOM from 'react-dom';
import Markdown from 'react-markdown';
import './css/main.css';
import AceEditor from 'react-ace';
import 'brace/mode/assembly_x86';
// Import a Theme (okadia, github, xcode etc)
import 'brace/theme/tomorrow_night_eighties';
import { JSONEditor } from 'react-json-editor-viewer';
import raymarcher_s from './asm/raymarcher.s';
import readme_md from './Readme.md';

function onChange(newValue) {
    console.log('change', newValue);
}

const _wasm = import("guppy_rust");

class TextEditorComponent extends React.Component {

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
        let def_value = raymarcher_s;
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

class ParametersComponent extends React.Component {

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

class ReadmeComponent extends React.Component {

    constructor(props, context) {
        super(props, context);
    }

    componentDidMount() {
    }

    render() {

        return (
            <Markdown className="Markdown" source={readme_md} />
        );
    }
}

class MemoryComponent extends React.Component {

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
        this.neededWidth = 4 * 1024;
        this.neededHeight = 1024;
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
        if (this.globals().active_mask_history) {
            this.neededWidth = this.globals().active_mask_history.length + 512;
            this.neededHeight = (this.globals().gpuConfig["wave_size"] + 1) *
                this.globals().gpuConfig["CU_count"] * this.globals().gpuConfig["waves_per_cu"] + 3 * 512;
        }
        this.draw = false;
        this.canvas.width = this.neededWidth;
        this.canvas.height = this.neededHeight;
        this.ctx.fillStyle = "#222222";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        var x = 0;
        var y = 0;
        {
            var canvas = this.ctx;
            canvas.font = "14px Monaco, monospace";
            var welcomeMessage = "(white, black, grey, blue, red) = (active, inactive, disabled, stalled, not enough resources)";
            canvas.textAlign = "start";
            canvas.textBaseline = "top";
            canvas.fillStyle = "#ffffff";
            canvas.fillText(welcomeMessage, x, y);
            y += 16;
        }
        if (this.globals().active_mask_history) {
            let history = this.globals().active_mask_history;
            if (history.length > 0) {
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "exec mask history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, x, y);
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
        {
            var canvas = this.ctx;
            canvas.font = "14px Monaco, monospace";
            var welcomeMessage = "(r, g, b) = (hits, misses, evictions)";
            canvas.textAlign = "start";
            canvas.textBaseline = "top";
            canvas.fillStyle = "#ffffff";
            canvas.fillText(welcomeMessage, 0, y + 16);
            y += 16;
        }
        if (this.globals().l2_metric_history) {
            let history = this.globals().l2_metric_history;
            if (history.length > 0) {

                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "L2$";
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
                var welcomeMessage = "Samplers $";
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
            "group_size": 32, "groups_count": 2048, "cycles_per_iter": 1
        };
        this.globals.gpuConfig = {
            "DRAM_latency": 32,
            "DRAM_bandwidth": 12 * 64, "L1_size": 1024, "L1_latency": 4,
            "L2_size": 1 * 1024, "L2_latency": 16, "sampler_cache_size": 1 * 1024,
            "sampler_latency": 16, "VGPRF_per_pe": 128, "wave_size": 32,
            "CU_count": 16, "ALU_per_cu": 1, "waves_per_cu": 4, "fd_per_cu": 1,
            "ALU_pipe_len": 4
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
                content: [
                    {
                        type: 'column',
                        content: [
                            {
                                type: 'react-component',
                                component: 'TextEditor',
                                title: 'TextEditor',
                                props: { globals: () => this.globals }

                            },
                            {
                                type: 'row',
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
                    }
                    ,
                    {
                        type: 'stack',
                        content: [
                            {
                                type: 'react-component',
                                component: 'Canvas',
                                title: 'Canvas',
                                props: { globals: () => this.globals }

                            },
                            {
                                type: 'react-component',
                                component: 'Readme',
                                title: 'Readme',
                                props: { globals: () => this.globals }

                            },
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
            TextEditorComponent
        );
        layout.registerComponent('Parameters',
            ParametersComponent
        );
        layout.registerComponent('Memory',
            MemoryComponent
        );
        layout.registerComponent('Readme',
            ReadmeComponent
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