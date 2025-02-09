import { useEffect, useMemo, useRef } from "react";
import axios from "axios";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };

export const ScatterPlot = ({
  plotId,
  width,
  height,
  data,
  colorScale,
  setSpecData,
  globalTimestamp,
  setGlobalTimestamp,
  hoveredPlotId,
  setHoveredPlotId
}) => {
  const dataIndex = useRef(null);
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

  // Function to get color for a label
  function toColor(label) {
    return colorScale(label);
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


  const onMouseOverCircle = (e, plotId) => {
    const circle = e.currentTarget;   
    const hoveredIndex = parseInt(circle.getAttribute("data-index"), 10);
    const hoveredTimestamp = data.timestamps[hoveredIndex];

    setHoveredPlotId(plotId);  // Track which plot is actively hovered
    setGlobalTimestamp(hoveredTimestamp); // Sync timestamp for other plots
    dataIndex.current = hoveredIndex; // Set index only for hovered plot
    console.log("Hovered index:", hoveredIndex);
};
const onMouseLeavePlot = () => {
  setHoveredPlotId(null); // Allow normal syncing again
};


useEffect(() => {
  if (globalTimestamp === null) return;
  
  // Ensure plotId is properly passed and avoid updating hovered plot
  if (plotId !== undefined && plotId === hoveredPlotId) return;
  
  // Find the closest timestamp in this plot's data
  const closestIndex = data.timestamps.reduce((bestIdx, ts, idx) => 
    Math.abs(ts - globalTimestamp) < Math.abs(data.timestamps[bestIdx] - globalTimestamp) ? idx : bestIdx, 
  0
);

  dataIndex.current = closestIndex; // update dataIndex of current plot
  console.log("Closest index:", closestIndex);
}, [globalTimestamp]); // Only update when globalTimestamp changes
  
  const handleClick = (event, index) => {
    console.log("Clicked on circle:", index);
    const dataPoint = {
      'x': data.x[index],
      'y': data.y[index],
      'z': data.time_within_file[index],
      'source_file': data.audio_filenames[index],
      'meta': data.metadata,
      'index': index,
      'label': data.label[index]
    };
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

  const points = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= data.x.length; i++) {
      pts.push(
        <circle
          key={i}
          data-index={i}  // Use `i` directly
          r={2} // radius
          cx={xScale(data.x[i])} // position on the X axis
          cy={yScale(data.y[i])} // on the Y axis
          opacity={1}
          stroke={toColor(data.labels.ground_truth[i])} // Apply correct color
          fill={toColor(data.labels.ground_truth[i])}  // Fill with the same color
          fillOpacity={0.2}
          strokeWidth={1}
          pointerEvents="all" // Ensure the element can be clicked
          onMouseEnter={(e) => onMouseOverCircle(e, plotId)}
        />
      );
    }
    // console.log("current point:", dataIndex.current);
    return pts;
  }, [data, xScale, yScale]);
      
  // const Cursor = ({ x, y, color, index }) => {
  const Cursor = ({ index, data }) => {
    const x = xScale(data.x[index]);
    const y = yScale(data.y[index]);
    const color = toColor(data.label[index]);

    const time_within_file = data.time_within_file[index];
    const source_file = data.audio_filenames[index];
    const time_accum = data.timestamps[index];

  
    const width =  50;
    const height = 50;
    // console.log("Cursor:", x, y, color, index);
    return (
      <>
        <circle 
          cx={x} 
          cy={y} 
          r={3} 
          // fill={color}
          fill="black"
          onClick={(e) => handleClick(e, index)}
        />
        {/* <rect 
          x={x-width} 
          y={y-height} 
          width={width} 
          height={height} 
          fill="#AAAAAA"
          visibility={'visible'}></rect> */}
        {/* <text 
          x={x-width+2} 
          y={y-height+12} 
          fontFamily="Verdana" 
          fontSize="12" 
          fill="white">{index}</text> */}
          <text 
            // x={width - 30} 
            y={height + 270} 
            fontFamily="Verdana" 
            fontSize="12" 
            fill="black"
          >
            <tspan x={width - 20} dy="1.2em">
              Time within file: {time_within_file.toFixed(2)}
            </tspan>
            <tspan x={width - 20} dy="1.2em">
              Source file: {source_file}
            </tspan>
            <tspan x={width - 20} dy="1.2em">
              Time accum: {time_accum.toFixed(2)}
            </tspan>
          </text>

      </>
    );
  };

  return (
    <div>
      <svg 
        width={width} 
        height={height} 
        style={{ pointerEvents: "all" }}
        onMouseLeave={onMouseLeavePlot}
      >

        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        >
          {points}
          {dataIndex.current && (
            <Cursor
              index={dataIndex.current}
              data={data}
            />
          )}
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
