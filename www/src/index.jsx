import GoldenLayoutWrapper from './app';
import { createStore } from 'redux';
import { Provider } from 'react-redux';
import reducer from './reducer';
import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import { HashRouter } from 'react-router-dom';

function setState(state) {
    return {
        type: 'SET_STATE',
        state
    };
}
const store = createStore(reducer);
store.dispatch(setState({ 'count': 10 }));

ReactDOM.render(
    <HashRouter>
        <Provider store={store}>
            <GoldenLayoutWrapper />
        </Provider>
    </HashRouter>,
    document.getElementById('root')
);



