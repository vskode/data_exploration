import { useEffect, useMemo, useRef } from "react";
import axios from "axios";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };
var PairingVariable = null;

export const LineChart = ({
  width,
  height,
  data,
  cursorPosition,
  setCursorPosition,
  color,
}) => {
  // bounds = area inside the graph axis = calculated by substracting the margins
  const axesRef = useRef(null);
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const [min, yMax] = d3.extent(data.y);
  const yScale = useMemo(() => {
    return d3
    .scaleLinear()
    .domain([0, yMax || 0])
    .range([boundsHeight, 0]);
  }, [data, height]);

  // X axis
  const [xMin, xMax] = d3.extent(data.x);
  const xScale = useMemo(() => {
    return d3
      .scaleLinear()
      .domain([0, xMax || 0])
      .range([0, boundsWidth]);
  }, [data, width]);

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
  const getClosestPoint = (cursorPixelPosition) => {
    const x = xScale.invert(cursorPixelPosition);
    let minDistance = Infinity;
    let closest = null;
    
    if (PairingVariable == null){
      let i = 0;
      for (const point of data.x) {
        const xDist = Math.abs(point - x)/xMax;
        const distance = xDist;
        if (distance < minDistance) {
          minDistance = distance;
          // closest = point;
          closest = {
            'x': data.x[i],
            'y': data.y[i],
            'z': data.z[i],
          };
          PairingVariable = closest['z']
        }
        i++
      }
    }
    else {
      let index = data.z.findIndex((e) => e == PairingVariable)
      closest = {
        'x': data.x[index],
        'y': data.y[index],
        'z': data.z[index],
      };
    }
    return closest;
  };
      
  //
  const onMouseMove = (e) => {
    PairingVariable = null;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;

    const closest = getClosestPoint(mouseX);

    setCursorPosition(xScale(closest.x));
  };
  
  const handleClick = (event, dataPoint) => {
    event.stopPropagation();  // Prevent event from being swallowed by other elements
    console.log("Circle clicked:", dataPoint);
    const url = "http://127.0.0.1:8000/";
    axios.post(url+'getDataPoint/', dataPoint)
    .then(response => {
      console.log(response.data)
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
      stroke="#cb1dd1"
      fill="#ABABAB"
      fillOpacity={0.2}
      strokeWidth={1}
    />         
    )
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
              color={color}
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
            onClick={(e) => handleClick(e, getClosestPoint(cursorPosition))}
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

  const width = 150;
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
