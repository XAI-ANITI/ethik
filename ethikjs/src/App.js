import React, { Component } from 'react';
import './App.css';
import BarChart from './BarChart.js'

class App extends Component {

  state = {
    data: [12, 5, 6, 6, 9, 14],
    width: 1400,
    height: 500
  }

  render() {
    return (
      <div className="App">
        <BarChart data={this.state.data} width={this.state.width} height={this.state.height} />
      </div>
    );
  }
}

export default App;
