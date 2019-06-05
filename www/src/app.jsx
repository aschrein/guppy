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

    onChange(newValue) {
        this.text = newValue;
    }
    onClick() {
        if (this.props.globals()) {
           
            this.props.globals().wasm.guppy_dispatch(
                this.text, 32, 1
            );
            this.props.globals().run = true;
        } else {
            console.log("[WARNING] wasm in null");
        }
    }
    render() {
        return (
            <div>
                <button onClick={this.onClick}>
                    Execute
                </button>
                <AceEditor
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

class GoldenLayoutWrapper extends React.Component {
    timer() {
        if (this.globals.wasm && this.globals.run) {
            if (!this.globals.wasm.guppy_clock()) {
                this.globals.run = false;
            } else {
                console.log(this.globals.wasm.guppy_get_active_mask());
            }
        }
    }
    componentDidMount() {
        this.globals = {};
        this.globals.wasm = null;
        _wasm.then(wasm => {

            this.globals.wasm = wasm;
            this.globals.wasm.guppy_create_gpu_state();
            console.log("wasm loaded");

        });
        this.intervalId = setInterval(this.timer.bind(this), 1);
        // Build basic golden-layout config
        const config = {
            content: [{
                type: 'row',
                content: [{
                    type: 'react-component',
                    component: 'TestComponentContainer'
                }, {
                    type: 'react-component',
                    component: 'IncrementButtonContainer'
                }, {
                    type: 'react-component',
                    component: 'DecrementButtonContainer'
                }]
            }]
        };

        function wrapComponent(Component, globals) {
            class Wrapped extends React.Component {
                render() {
                    return (
                        <Component globals={globals} />
                    );
                }
            }
            return Wrapped;
        };

        var layout = new GoldenLayout(config, this.layout);
        layout.registerComponent('IncrementButtonContainer',
            wrapComponent(App, () => this.globals)
        );
        layout.registerComponent('DecrementButtonContainer',
            wrapComponent(App, () => this.globals)
        );
        layout.registerComponent('TestComponentContainer',
            wrapComponent(App, () => this.globals)
        );
        layout.init();

        window.addEventListener('resize', () => {
            layout.updateSize();
        });
        window.React = React;
        window.ReactDOM = ReactDOM;
    }

    render() {
        return (
            <div className='goldenLayout' ref={input => this.layout = input} />
        );
    }
}


export default GoldenLayoutWrapper;