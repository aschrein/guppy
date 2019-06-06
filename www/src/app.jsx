import GoldenLayout from 'golden-layout';
import React from 'react';
import ReactDOM from 'react-dom';
import './css/main.css';
import AceEditor from 'react-ace';
import 'brace/mode/assembly_x86';

// Import a Theme (okadia, github, xcode etc)
import 'brace/theme/twilight';

function onChange(newValue) {
    console.log('change', newValue);
}

const _wasm = import("guppy_rust");

class App extends React.Component {

    constructor(props, context) {
        super(props, context);

        this.onChange = this.onChange.bind(this);
        this.onClick = this.onClick.bind(this);
    }

    componentDidMount() {
        // this.refs.editor.setValue(
        //     "ret"
        // );
    }

    onChange(newValue) {
        this.text = newValue;
    }
    onClick() {
        if (this.props.globals().wasm) {
            this.props.globals().active_mask_history = [];
            this.props.globals().wasm.guppy_dispatch(
                this.text, 32, 8
            );
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
            <div>
                <button onClick={this.onClick}>
                    Execute
                </button>
                <AceEditor
                    value={def_value}
                    ref="editor"
                    mode="assembly_x86"
                    theme="twilight"
                    onChange={this.onChange}
                    name="UNIQUE_ID_OF_DIV"
                    editorProps={{
                        $blockScrolling: true
                    }}
                />
            </div>
        );
    }
}

class CanvasComponent extends React.Component {
    componentDidMount() {
        this.updateCanvas = this.updateCanvas.bind(this);
        this.onResize = this.onResize.bind(this);
        this.ctx = this.refs.canvas.getContext('2d');
        this.canvas = this.refs.canvas;
        this.props.glContainer.on('resize', this.onResize);
        this.globals = this.props.globals;
        this.globals().updateCanvas = this.updateCanvas;

        this.updateCanvas();

        // layout.on('componentCreated',function(component) {
        //     component.container.on('resize',function() {
        //         console.log('component.resize', component);
        //         if (component.onResize) {
        //             component.onResize();
        //         }
        //     });
        // });
        // this.props.container.on('resize', function (comp) {
        //     console.log('component.resize', comp.componentName);
        // });
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.updateCanvas)
    }

    onResize() {
        this.updateCanvas();
    }

    updateCanvas() {
        const width = this.props.glContainer.width;
        const height = this.props.glContainer.height;
        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx.fillStyle = "#222222";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.globals().active_mask_history) {
            let history = this.globals().active_mask_history;
            if (history.length > 0) {
                var canvas = this.ctx;
                canvas.font = "14px Monaco, monospace";
                var welcomeMessage = "exec mask history";
                canvas.textAlign = "start";
                canvas.textBaseline = "top";
                canvas.fillStyle = "#ffffff";
                canvas.fillText(welcomeMessage, 0, 0);
                // console.log('updateCanvas', history[0].length);
                var x = 0;
                var y = 16;
                for (var i = 0; i < history.length; i++) {
                    x = 0;
                    for (var j = 0; j < history[0].length; j++) {
                        if (j % 32 == 0)
                            x += 4;
                        if (history[i][j] == 1) {
                            this.ctx.fillStyle = "white";
                        } else if (history[i][j] == 0) {
                            this.ctx.fillStyle = "black";
                        } else if (history[i][j] == 2) {
                            this.ctx.fillStyle = "grey";
                        }
                        this.ctx.fillRect(x, y, 2, 2);
                        x += 2;
                    }
                    y += 2;
                }
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
                //if (this.globals.updateCanvas)
                this.globals.updateCanvas();
                // console.log();
            }
        }
    }
    componentDidMount() {
        this.globals = {};
        this.globals.wasm = null;
        this.globals.active_mask_history = null;
        
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

                }
                    // , {
                    //     type: 'react-component',
                    //     component: 'Dummy',
                    //     title: 'Dummy',
                    //     props: { globals: () => this.globals }
                    // }
                ]
            }]
        };

        function wrapComponent(Component, container, globals) {
            class Wrapped extends React.Component {
                // componentDidMount() {
                //     this.onResize = this.onResize.bind(this);
                //     this.component = this.refs.component;
                // }
                // onResize() {
                //     this.component.onResize();
                // }
                render() {
                    return (
                        <Component refs="component" container={container} globals={globals} />
                    );
                }
            }
            return Wrapped;
        };

        var layout = new GoldenLayout(config, this.layout);
        this.layout = layout;
        let globals = this.globals;
        _wasm.then(wasm => {
            layout.updateSize();
            this.globals.wasm = wasm;
            this.globals.wasm.guppy_create_gpu_state();
            this.globals.active_mask_history = [];
            console.log("wasm loaded");

        });
        layout.registerComponent('Canvas', CanvasComponent
        );
        layout.registerComponent('TextEditor',
            App
        );
        // layout.registerComponent('Dummy',
        //     App
        // );
        layout.init();
        window.React = React;
        window.ReactDOM = ReactDOM;
        window.addEventListener('resize', () => {
            layout.updateSize();
        });


        layout.updateSize();
    }

    render() {
        return (
            <div className='goldenLayout' ref={input => this.layout = input} />
        );
    }
}


export default GoldenLayoutWrapper;