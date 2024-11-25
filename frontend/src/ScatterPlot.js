import { useEffect, useMemo, useRef } from "react";
import axios from "axios";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };
var PairingVariable = null;

export const ScatterPlot = ({
  width,
  height,
  data,
  setSpecData,
  cursorPosition,
  setCursorPosition,
  color,
}) => {
  // bounds = area inside the graph axis = calculated by substracting the margins
  const axesRef = useRef(null);
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;
  
  const [yMin, yMax] = d3.extent(data.y);
  const yScale = useMemo(() => {
    return d3
    .scaleLinear()
    .domain([yMin, yMax || 0])
    .range([boundsHeight, 0]);
  }, [data, height]);

  // X axis
  const [xMin, xMax] = d3.extent(data.x);
  const xScale = useMemo(() => {
    return d3
      .scaleLinear()
      .domain([xMin, xMax || 0])
      .range([0, boundsWidth]);
  }, [data, width]);
  
  // t axis
  const [tMin, tMax] = d3.extent(data.timestamp);
  const tScale = useMemo(() => {
    return d3
      .scaleLinear()
      .domain([tMin, tMax || 0]);
  }, [data, width]);

  function onlyUnique(value, index, array) {
    return array.indexOf(value) === index;
  }
  const labels = data.label.filter(onlyUnique).sort();
  const length = labels.length;
  const colors = Array.from({length: length}, (_, n) => n*(tMax/(length-1)))
  const col_dict = {}
  for (let i = 0; i < length; i++) {
    col_dict[labels[i]] = colors[i]
  }
  if (data.label === undefined) {
    data.label = Array.from({length: data.x.length}, (_, n) => n);
  }

  // Render the X and Y axis using d3.js, not react
  useEffect(() => {
    const svgElement = d3.select(axesRef.current);
    svgElement.selectAll("*").remove();
    const xAxisGenerator = d3.axisBottom(xScale);
    svgElement
      .append("g")
      .attr("transform", "translate(0," + boundsHeight + ")")
      .call(xAxisGenerator);

    const yAxisGenerator = d3.axisLeft(yScale);
    svgElement.append("g").call(yAxisGenerator);
  }, [xScale, yScale, boundsHeight]);

  //
  const getClosestPoint = (cursorPixelPositionX, cursorPixelPositionY) => {
    const x = xScale.invert(cursorPixelPositionX);
    const y = yScale.invert(cursorPixelPositionY);
    const a = data.x.map(e => Math.pow(e - x, 2))
    const b = data.y.map(e => Math.pow(e - y, 2))
    const argFact = (compareFn) => (array) => array.map((el, idx) => [el, idx]).reduce(compareFn)[1]
    const argMin = argFact((max, el) => (el[0] < max[0] ? el : max))
    const dists = a.map((e, i) => e+b[i])
    const in_close = argMin(dists)
    let minDistance = Infinity;
    let closest = null;
    if (PairingVariable == null){
      closest = {
        'x': data.x[in_close],
        'y': data.y[in_close],
        'z': data.timestamp[in_close],
        'meta': data.metadata,
        'index': in_close,
        // 'label': data.label[in_close]
      };
      PairingVariable = closest['z']
    }
    else {
      let index = argMin(data.timestamp.map((e, i) => Math.abs(e-PairingVariable)))
      PairingVariable = data.timestamp[index]
      closest = {
        'x': data.x[index],
        'y': data.y[index],
        'z': data.timestamp[index],
        'meta': data.metadata,
        'index': index,
        // 'label': data.label[in_close]
      };
    }
    return closest;
  };
      
  //
  const onMouseMove = (e) => {
  // const onMouseOver = (e) => {
    PairingVariable = null;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const closest = getClosestPoint(mouseX, mouseY);

    setCursorPosition([xScale(closest.x), yScale(closest.y)]);
  };

  
  const handleClick = (event) => {
    const dataPoint = getClosestPoint(cursorPosition);
    event.stopPropagation();  // Prevent event from being swallowed by other elements
    console.log("Circle clicked:", dataPoint);
    const url = "http://127.0.0.1:8000/";
    axios.post(url+'getDataPoint/', dataPoint)
    .then(response => {
      console.log(response.data)
      setSpecData(response.data.spectrogram_data)
    })
    .catch(function (error) {
      // handle error
      console.log(error);
    })
  };

  const points = [];
  for (let i = 0; i <= data.x.length; i++){
    points.push(
    <circle
      key={i}
      r={4} // radius
      cx={xScale(data.x[i])} // position on the X axis
      cy={yScale(data.y[i])} // on the Y axis
      opacity={1}
      // stroke={toColor(data.timestamp[i])}
      stroke={toColor(data.label[i])}
      fill="#ABABAB"
      fillOpacity={0.2}
      strokeWidth={1}
      // onMouseOver={Cursor(xScale(data.x[i]), xScale(data.y[i]), color)}
      // onMouseOver={onMouseOver}
      // onMouseOver={Cursor(i, color)}
    />         
    )
  }


  function toColor(num) {
    let col = 0
    // if (num instanceof String) {
    col = tScale(col_dict[num]);
    // } else {
    //   col = tScale(num);
    // }
    const c = d3['interpolateViridis'](col);
    return c;
}

  return (
    <div>
      <svg width={width} height={height}>
        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        >
          {points}
          {cursorPosition && (
            <Cursor
              height={boundsHeight}
              x={xScale(getClosestPoint(cursorPosition)?.x)}
              y={yScale(getClosestPoint(cursorPosition)?.y)}
              color={toColor(getClosestPoint(cursorPosition)?.z)}
            />
          )}
          <rect
            x={0}
            y={0}
            width={boundsWidth}
            height={boundsHeight}
            onMouseMove={onMouseMove}
            onMouseLeave={() => setCursorPosition(null)}
            visibility={"hidden"}
            pointerEvents={"all"}
            // onClick={(e) => handleClick(e, getClosestPoint(cursorPosition))}
            onClick={handleClick}
          />
        </g>
        <g
          width={boundsWidth}
          height={boundsHeight}
          ref={axesRef}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        />
      </svg>
    </div>
  );
}

const Cursor = ({ x, y, color }) => {

  const width =  50;
  const height = 50;

  return (
    <>
      <circle 
        cx={x} 
        cy={y} 
        r={5} 
        fill={color}
      />
      <rect 
        x={x-width} 
        y={y-height} 
        width={width} 
        height={height} 
        fill="#AAAAAA"
        visibility={'visible'}></rect>
      <text 
        x={x-width+2} 
        y={y-height+12} 
        fontFamily="Verdana" 
        fontSize="12" 
        fill="white">{PairingVariable}</text>
    </>
  );
};