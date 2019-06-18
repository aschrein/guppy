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
import branch_1_s from './asm/branch_1.s';
import branch_2_s from './asm/branch_2.s';
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
        this.setText = this.setText.bind(this);
        this.setClocks = this.setClocks.bind(this);
        this.state = { clocks: 0 };

    }

    componentDidMount() {
        // this.refs.editor.setValue(
        //     "ret"
        // );
        this.props.glContainer.on('resize', this.onResize);
        this.props.globals().setText = this.setText;
        this.props.globals().setClocks = this.setClocks;
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
    setClocks(clocks) {
        this.setState({ clocks: clocks });
    }
    setText(text) {
        this.refs.editor.editor.setValue(text);
        console.log(this.refs.editor);
    }

    Execute() {
        let globals = this.props.globals();
        if (this.props.globals().wasm) {
            this.props.globals().resetGPU();
            let config = this.props.globals().dispatchConfig;
            for (var i = 0; i < globals.r_images.length; i++) {
                globals.wasm.guppy_put_image(globals.r_images[i]);
            }
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

        return (
            <div className="ace_editor_container">
                <button style={{ margin: 10 }} onClick={this.Execute}>
                    Execute
                </button>
                <button style={{ margin: 10 }} onClick={this.PauseResume}>
                    Pause/Resume Clocks:{this.state.clocks}
                </button>
                <div style={{ background: "#cccccc", margin: 10 }}>
                    <p style={{ margin: 5 }}>Setup example</p>
                    <button style={{ border: 1 }} onClick={this.props.globals().setupRaymarching}>
                        RayMarching
                    </button>
                    <button style={{ border: 1 }} onClick={this.props.globals().setupBranch_1}>
                        branch_1
                    </button>
                    <button style={{ border: 1 }} onClick={this.props.globals().setupBranch_2}>
                        branch_2
                    </button>
                </div>
                <AceEditor
                    value={this.text}
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
        this.updateParameters = this.updateParameters.bind(this);
    }

    componentDidMount() {
        this.props.globals().updateParameters = this.updateParameters;
    }

    onChange(key, value, parent, data) {
        console.log("onchange", key, value);
        this.props.globals().gpuConfig[key] = value;


    }

    onChangeDispatch(key, value, parent, data) {
        console.log("onChangeDispatch", key, value);
        this.props.globals().dispatchConfig[key] = value;

    }

    updateParameters() {
        // console.log(this.refs.gpu_config.state);
        this.refs.gpu_config.setState({ data: { root: this.props.globals().gpuConfig } });
        this.refs.dispatch_config.setState({ data: { root: this.props.globals().dispatchConfig } });
        // this.forceUpdate();
        // this.refs.gpu_config.setData(this.props.globals().gpuConfig);
        // this.refs.dispatch_config.setData(this.props.globals().dispatchConfig);
    }

    render() {
        // console.log(this.props.globals().gpuConfig);
        return (
            <div>
                <p style={{ color: "white", margin: 10 }}>GPU Config</p>
                <JSONEditor
                    ref="gpu_config"
                    data={
                        this.props.globals().gpuConfig
                    }
                    onChange={this.onChange}
                />
                <p style={{ color: "white", margin: 10 }}>Dispatch config</p>
                <JSONEditor
                    ref="dispatch_config"
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

class BindingsComponent extends React.Component {

    constructor(props, context) {
        super(props, context);

        this.updateMemory = this.updateMemory.bind(this);
    }

    componentDidMount() {
        this.props.globals().updateMemory = this.updateMemory;
        this.ctx = this.refs.canvas.getContext('2d');
        let promiseOnload = (img) => {
            return new Promise(resolve => {

                img.onload = () => {
                    // console.log("images Promise");
                    resolve({ img, status: 'ok' });
                };
                img.onerror = () => resolve({ img, status: 'error' });
            });
        }
        let t0 = this.refs.t0;
        let t1 = this.refs.t1;
        Promise.all([promiseOnload(t0), promiseOnload(t1)]).then((value) => {
            // console.log("images loaded");
            let pushImg = (img) => {
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
            }
            pushImg(t0);
            pushImg(t1);
            this.props.globals().resetGPU();
            this.updateMemory();
            // console.log(base64);
        });

    }

    updateMemory() {
        let ctx = this.refs.canvas.getContext('2d');
        let image = new Image();
        let canvas = this.refs.canvas;

        let globals = this.props.globals();

        let t0 = this.refs.t0;
        let t1 = this.refs.t1;
        image.onload = function () {
            canvas.width = 512;
            canvas.height = 1024;
            // canvas.width = image.width;
            // canvas.height += image.height;
            ctx.fillStyle = "#222222";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            var x = 0;
            var y = 5;
            let drawText = (text) => {

                ctx.font = "14px Monaco, monospace";
                ctx.textAlign = "start";
                ctx.textBaseline = "top";
                ctx.fillStyle = "#ffffff";
                ctx.fillText(text, x, y);
                // y += 16;
                x += text.length * 8;
            };
            drawText("s0 = Sampler {WrapMode::WRAP, SampleMode::BILINEAR}");
            y += 16;
            x = 0;
            drawText("t0 = ");
            ctx.drawImage(t0, x, y);
            y += t0.height + 10;
            x = 0;

            drawText("t1 = ");
            ctx.drawImage(t1, x, y);
            y += t1.height + 10;
            x = 0;
            drawText("u0 = ");
            ctx.drawImage(image, x, y);
        };
        let base64 = "data:image/png;base64," + globals.wasm.guppy_get_image(0, false);
        // console.log(base64);
        image.src = base64;
    }

    render() {

        return (
            <div>
                <canvas ref="canvas" />
                <img style={{ display: "none" }} ref="t0" src="img/lenna.png"></img>
                <img style={{ display: "none" }} ref="t1" src="img/rhino.png"></img>
            </div>
        );
    }
}

class GraphsComponent extends React.Component {

    constructor(props, context) {
        super(props, context);
        this.neededWidth = 4 * 1024;
        this.neededHeight = 1024;
        this.updateCanvas = this.updateCanvas.bind(this);
        this.onResize = this.onResize.bind(this);
        this.scheduleDraw = this.scheduleDraw.bind(this);
        this.remapColors = this.remapColors.bind(this);
        this.colorMap = {};
        this.getRandomColor = () => {
            var letters = '0123456789ABCDEF';
            var color = '#';
            for (var i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
            }
            return color;
        };
    }

    componentDidMount() {
        this.draw = true;
        this.ctx = this.refs.canvas.getContext('2d');
        this.canvas = this.refs.canvas;
        this.lastClock = 0;
        this.props.glContainer.on('resize', this.onResize);
        this.globals = this.props.globals;
        this.globals().updateCanvas = this.scheduleDraw;

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

    remapColors() {

        let waves_per_cu = this.globals().gpuConfig["waves_per_cu"];
        for (var wave_id = 0; wave_id < waves_per_cu; wave_id++) {

            this.colorMap[wave_id] = this.getRandomColor();
        }
        this.draw = true;
        this.updateCanvas();
    }

    updateCanvas() {
        if (!this.draw) {
            return;
        }

        let wave_size = this.globals().gpuConfig["wave_size"];
        let waves_per_cu = this.globals().gpuConfig["waves_per_cu"];
        let cu_count = this.globals().gpuConfig["CU_count"];
        let valu_count = this.globals().gpuConfig["ALU_per_cu"];
        let valu_pipe_len = this.globals().gpuConfig["ALU_pipe_len"];
        if (this.globals().active_mask_history) {
            this.neededWidth = this.globals().active_mask_history.length + 512;
            this.neededHeight = (wave_size + 4 + valu_count * 4 + 14) *
                this.globals().gpuConfig["CU_count"] * this.globals().gpuConfig["waves_per_cu"] + 3 * 512;
        }
        this.draw = false;
        this.canvas.width = this.neededWidth;
        this.canvas.height = this.neededHeight;
        this.ctx.fillStyle = "#222222";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        var x = 0;
        var y = 0;
        let putTag = (text, x, y) => {
            var canvas = this.ctx;
            canvas.font = "10px Monaco, monospace";
            canvas.textAlign = "start";
            canvas.textBaseline = "top";
            canvas.fillStyle = "#ffffff";
            canvas.fillText(text, x, y);
        };
        let color_sem = {
            0: "inactive",
            1: "active",
            2: "disabled",
            3: "stalled",
            4: "idle"
        };
        let color_code = {
            0: "black",
            1: "white",
            2: "grey",
            3: "blue",
            4: "red"
        };
        {
            //8:  4021
            //32: 4226
            //64: 4524

            var canvas = this.ctx;
            for (var i = 0; i < 5; i++) {
                this.ctx.fillStyle = color_code[i];
                this.ctx.fillRect(x, y, 8, 8);
                canvas.font = "14px Monaco, monospace";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(color_sem[i], x + 8, y);
                y += 12;
            }
        }
        let HISTORY_OFFSET = 48;
        if (this.globals().active_mask_history) {
            var sampler_max_metrics = Array(cu_count).fill(0);
            for (var i = 0; i < this.globals().sampler_cache_history.length; i++) {
                let row = this.globals().sampler_cache_history[i];
                for (var j = 0; j < cu_count; j++) {
                    sampler_max_metrics[j] = Math.max(sampler_max_metrics[j], row[j * 3 + 0], row[j * 3 + 1], row[j * 3 + 2]);
                }
            }
            let history = this.globals().active_mask_history;
            if (history.length > 0) {
                putTag("Clocks:" + this.globals().clocks, 0, y);
                var canvas = this.ctx;
                // console.log('updateCanvas', history[0].length);
                x = HISTORY_OFFSET;
                var exec_mask_offset = y + 16;
                for (var i = 0; i < history.length; i++) {
                    this.neededHeight = Math.max(this.neededHeight, y);
                    y = exec_mask_offset + 4;
                    for (var cu_id = 0; cu_id < cu_count; cu_id++) {
                        y += 8;
                        if (i == 0)
                            putTag("CU#" + cu_id, 0, y);
                        y += 12;
                        for (var wave_id = 0; wave_id < waves_per_cu; wave_id++) {
                            if (i == 0) {
                                putTag("wave#" + wave_id, 0, y + wave_size / 2);
                                if (!(wave_id in this.colorMap))
                                    this.colorMap[wave_id] = this.getRandomColor();
                                this.ctx.fillStyle = this.colorMap[wave_id];
                                this.ctx.fillRect(38, y + wave_size / 2, 4, 4);
                            }

                            for (var lane_id = 0; lane_id < wave_size; lane_id++) {
                                let j = cu_id * waves_per_cu * wave_size + wave_id * wave_size + lane_id;
                                if (j % wave_size == 0)
                                    y += 4;
                                this.ctx.fillStyle = color_code[history[i][j]];
                                this.ctx.fillRect(x, y, 1, 1);
                                y += 1;
                            }
                        }
                        y += 4;
                        {
                            let history = this.globals().alu_active_history;
                            for (var valu_id = 0; valu_id < valu_count; valu_id++) {
                                if (i == 0)
                                    putTag("ALU#" + valu_id, 0, y);
                                let j = cu_id * valu_count + valu_id;
                                if (history[i][j] == -1) {
                                    this.ctx.fillStyle = "white";
                                } else if (history[i][j] == -2) {
                                    this.ctx.fillStyle = "black";
                                } else {
                                    this.ctx.fillStyle = this.colorMap[history[i][j]];
                                }
                                this.ctx.fillRect(x, y, 1, 6);
                                y += 14;
                            }
                        }
                        y += 4;
                        {
                            let history = this.globals().sampler_cache_history;
                            let hit = 10.0 * history[i][cu_id * 3 + 0] / sampler_max_metrics[cu_id];
                            let miss = 10.0 * history[i][cu_id * 3 + 1] / sampler_max_metrics[cu_id];
                            let evict = 10.0 * history[i][cu_id * 3 + 2] / sampler_max_metrics[cu_id];
                            if (i == 0)
                                putTag("sampler", 0, y);
                            for (var j = 10; j >= 0; j--) {
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
                        }
                    }
                    x += 1;
                    this.neededWidth = Math.max(this.neededWidth, x);
                }
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
                x = HISTORY_OFFSET;
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
            <div style={{ "marginLeft": 0, "marginRight": "40%" }}>
                <button onClick={this.scheduleDraw}>
                    Render History
            </button>
                <button onClick={this.remapColors}>
                    Shuffle Colors
            </button>
            </div>
            <canvas ref="canvas" /> </div>;
    }
}
class GoldenLayoutWrapper extends React.Component {
    constructor(props, context) {
        super(props, context);
        this.setupRaymarching = this.setupRaymarching.bind(this);
        this.setupBranch_1 = this.setupBranch_1.bind(this);
        this.setupBranch_2 = this.setupBranch_2.bind(this);
    }
    setupRaymarching() {
        this.globals.dispatchConfig = {
            "group_size": 32, "groups_count": 2048, "cycles_per_iter": 1, "update graph": false
        };
        this.globals.gpuConfig = {
            "DRAM_latency": 32,
            "DRAM_bandwidth": 12 * 64, "L1_size": 1024, "L1_latency": 4,
            "L2_size": 1 * 1024, "L2_latency": 16, "sampler_cache_size": 1 * 1024,
            "sampler_latency": 16, "VGPRF_per_pe": 128, "wave_size": 32,
            "CU_count": 64, "ALU_per_cu": 4, "waves_per_cu": 4, "fd_per_cu": 4,
            "ALU_pipe_len": 1
        };
        this.globals.setText(raymarcher_s);
        this.globals.resetGPU();
    }

    setupBranch_1() {
        this.globals.dispatchConfig = {
            "group_size": 32, "groups_count": 6, "cycles_per_iter": 1, "update graph": true
        };
        this.globals.gpuConfig = {
            "DRAM_latency": 32,
            "DRAM_bandwidth": 12 * 64, "L1_size": 1024, "L1_latency": 4,
            "L2_size": 1 * 1024, "L2_latency": 16, "sampler_cache_size": 1 * 1024,
            "sampler_latency": 16, "VGPRF_per_pe": 128, "wave_size": 32,
            "CU_count": 2, "ALU_per_cu": 2, "waves_per_cu": 4, "fd_per_cu": 2,
            "ALU_pipe_len": 1
        };
        this.globals.setText(branch_1_s);
        this.globals.resetGPU();
    }

    setupBranch_2() {
        this.globals.dispatchConfig = {
            "group_size": 32, "groups_count": 6, "cycles_per_iter": 1, "update graph": true
        };
        this.globals.gpuConfig = {
            "DRAM_latency": 32,
            "DRAM_bandwidth": 12 * 64, "L1_size": 1024, "L1_latency": 4,
            "L2_size": 1 * 1024, "L2_latency": 16, "sampler_cache_size": 1 * 1024,
            "sampler_latency": 16, "VGPRF_per_pe": 128, "wave_size": 32,
            "CU_count": 2, "ALU_per_cu": 2, "waves_per_cu": 4, "fd_per_cu": 2,
            "ALU_pipe_len": 1
        };
        this.globals.setText(branch_2_s);
        this.globals.resetGPU();
    }

    timer() {
        if (this.globals.wasm && this.globals.run) {
            for (var i = 0; i < this.globals.dispatchConfig["cycles_per_iter"]; i++) {
                if (!this.globals.wasm.guppy_clock()) {
                    this.globals.run = false;
                } else {
                    this.globals.clocks += 1;
                    this.globals.setClocks(this.globals.clocks);
                    this.globals.active_mask_history.push(this.globals.wasm.guppy_get_active_mask());
                    this.globals.alu_active_history.push(this.globals.wasm.guppy_get_valu_active());
                    this.globals.sampler_cache_history.push(this.globals.wasm.guppy_get_sampler_cache_metrics());
                    this.globals.l2_metric_history.push([
                        this.globals.wasm.guppy_get_gpu_metric("L2 hit"),
                        this.globals.wasm.guppy_get_gpu_metric("L2 miss"),
                        this.globals.wasm.guppy_get_gpu_metric("L2 evict"),
                    ]);
                    //if (this.globals.updateCanvas)
                    this.globals.updateMemory();
                    if (this.globals.dispatchConfig["update graph"])
                        this.globals.updateCanvas();
                }
            }
        }
    }
    componentDidMount() {
        this.globals = {};
        this.globals.dispatchConfig = {
            "group_size": 1, "groups_count": 1, "cycles_per_iter": 1, "update graph": true
        };
        this.globals.gpuConfig = {
            "DRAM_latency": 1,
            "DRAM_bandwidth": 1, "L1_size": 0, "L1_latency": 0,
            "L2_size": 1, "L2_latency": 1, "sampler_cache_size": 1,
            "sampler_latency": 1, "VGPRF_per_pe": 1, "wave_size": 1,
            "CU_count": 1, "ALU_per_cu": 1, "waves_per_cu": 1, "fd_per_cu": 1,
            "ALU_pipe_len": 1
        };
        this.globals.wasm = null;
        this.globals.clocks = 0;
        this.globals.setupBranch_1 = this.setupBranch_1;
        this.globals.setupBranch_2 = this.setupBranch_2;
        this.globals.setupRaymarching = this.setupRaymarching;
        this.globals.r_images = [];
        this.globals.active_mask_history = null;
        this.globals.sampler_cache_history = null;
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
                                type: 'stack',
                                content: [
                                    {
                                        type: 'react-component',
                                        component: 'Bindings',
                                        title: 'Bindings',
                                        props: { globals: () => this.globals }
                                    },
                                    {
                                        type: 'react-component',
                                        component: 'Parameters',
                                        title: 'Parameters',
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
                                component: 'Graphs',
                                title: 'Graphs',
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
            globals.wasm.guppy_init_framebuffer(256, 256);
            globals.active_mask_history = [];
            globals.alu_active_history = [];
            globals.sampler_cache_history = [];
            globals.l2_metric_history = [];
            globals.clocks = 0;
            if (globals.updateMemory)
                globals.updateMemory();
            if (globals.updateParameters)
                globals.updateParameters();
            if (globals.updateCanvas)
                globals.updateCanvas();
        }
        _wasm.then(wasm => {
            layout.updateSize();
            this.globals.wasm = wasm;
            this.setupBranch_1();

            //console.log("wasm loaded");
            //console.log(wasm.guppy_get_config());

        });
        layout.registerComponent('Graphs', GraphsComponent
        );
        layout.registerComponent('TextEditor',
            TextEditorComponent
        );
        layout.registerComponent('Parameters',
            ParametersComponent
        );
        layout.registerComponent('Bindings',
            BindingsComponent
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