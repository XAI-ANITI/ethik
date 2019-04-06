import React, {Component} from 'react';
import * as d3 from "d3";

class BarChart extends Component {
  componentDidMount() {
    this.drawChart();
  }

  drawChart() {

    var data = [
      {'x': 1, 'y': 20},
      {'x': 3, 'y': 30}
    ]

    var svgWidth = 600, svgHeight = 400;
    var margin = { top: 20, right: 20, bottom: 30, left: 50 };
    var width = svgWidth - margin.left - margin.right;
    var height = svgHeight - margin.top - margin.bottom;

    var svg = d3.select('body')
      .append('svg')
      .attr('width', svgWidth)
      .attr('height', svgHeight);

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    var x = d3.scaleLinear().rangeRound([0, width]);
    var y = d3.scaleLinear().rangeRound([height, 0]);

    var line = d3.line()
      .x(function(d) { return x(d.x)})
      .y(function(d) { return y(d.y)})
    x.domain(d3.extent(data, function(d) { return d.x }));
    y.domain(d3.extent(data, function(d) { return d.y }));

    // x-axis
    g.append("g")
       .attr("transform", "translate(0," + height + ")")
       .call(d3.axisBottom(x))
       .select(".domain")
       .remove();

    // y-axis
    g.append("g")
       .call(d3.axisLeft(y))
       .append("text")
       .attr("fill", "#000")
       .attr("transform", "rotate(-90)")
       .attr("y", 6)
       .attr("dy", "0.71em")
       .attr("text-anchor", "end")
       .text("Price ($)");

    // Line plot
    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 1.5)
      .attr("d", line);

    /*const data = this.props.data;
    const w = this.props.width;
    const h = this.props.height;

    const svg = d3.select("body")
    .append("svg")
    .attr("width", w)
    .attr("height", h)
    .style("margin-left", 100);

    svg.selectAll("rect")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", (d, i) => i * 70)
      .attr("y", (d, i) => h - 10 * d)
      .attr("width", 65)
      .attr("height", (d, i) => d * 10)
      .attr("fill", "green")*/
  }

  render() {
    return <div id={"#" + this.props.id}></div>
  }
}

export default BarChart;
