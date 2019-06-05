import GoldenLayout from 'golden-layout';
import {Provider} from 'react-redux';
import React from 'react';
import ReactDOM from 'react-dom';
import './css/main.css';
import AceEditor from 'react-ace';

import 'brace/mode/assembly_x86';

// Import a Theme (okadia, github, xcode etc)
import 'brace/theme/twilight';

function onChange(newValue) {
  console.log('change',newValue);
}


class App extends React.Component {

    constructor(props, context) {
        super(props, context);
        
        this.onChange = this.onChange.bind(this);
    }

    onChange(newValue) {
        console.log('change', newValue);
    }

    render() {
        return (
            <div>
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
    componentDidMount() {
        // Build basic golden-layout config
        const config = {
            content: [{
                type: 'row',
                content: [{
                    type: 'react-component',
                    component: 'TestComponentContainer'
                },{
                    type: 'react-component',
                    component: 'IncrementButtonContainer'
                },{
                    type: 'react-component',
                    component: 'DecrementButtonContainer'
                }]
            }]
        };

        function wrapComponent(Component, store) {
            class Wrapped extends React.Component {
                render() {
                    return (
                        <Provider store={store}>
                            <Component {...this.props}/>
                        </Provider>
                    );
                }
            }
            return Wrapped;
        };

        var layout = new GoldenLayout(config, this.layout);
        layout.registerComponent('IncrementButtonContainer', 
                                 wrapComponent(App, this.context.store)
        );
        layout.registerComponent('DecrementButtonContainer',
                                 wrapComponent(App, this.context.store)
        );
        layout.registerComponent('TestComponentContainer',
                                 wrapComponent(App, this.context.store)
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
            <div className='goldenLayout' ref={input => this.layout = input}/>
        );
    }
}

GoldenLayoutWrapper.contextTypes = {
    store: React.PropTypes.object.isRequired
};

export default GoldenLayoutWrapper;