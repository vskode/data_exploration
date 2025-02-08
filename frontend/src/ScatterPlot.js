import { useEffect, useMemo, useRef } from "react";
import axios from "axios";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };
var PairingVariable = null;

export const ScatterPlot = ({
  plotId,
  width,
  height,
  data,
  setSpecData,
  globalTimestamp,
  setGlobalTimestamp,
  hoveredPlotId,
  setHoveredPlotId,
  color,
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


  const onMouseOverCircle = (e, plotId) => {
    const circle = e.currentTarget;   
    const hoveredIndex = parseInt(circle.getAttribute("data-index"), 10);
    const hoveredTimestamp = data.timestamps[hoveredIndex];

    setHoveredPlotId(plotId);  // Track which plot is actively hovered
    setGlobalTimestamp(hoveredTimestamp); // Sync timestamp for other plots
    dataIndex.current = hoveredIndex; // Set index only for hovered plot
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
}, [globalTimestamp]); // Only update when globalTimestamp changes
  
  const handleClick = (event) => {
    const index = dataIndex.current;
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
          r={4} // radius
          cx={xScale(data.x[i])} // position on the X axis
          cy={yScale(data.y[i])} // on the Y axis
          opacity={1}
          stroke={toColor(data.label[i])}
          fill="#ABABAB"
          fillOpacity={0.2}
          strokeWidth={1}
          pointerEvents="all" // Ensure the element can be clicked
          onMouseEnter={(e) => onMouseOverCircle(e, plotId)}
        />
      );
    }
    return pts;
  }, [data, xScale, yScale]);
      
  const Cursor = ({ x, y, color, index }) => {
  
    const width =  50;
    const height = 50;
    // console.log("Cursor:", x, y, color, index);
    return (
      <>
        <circle 
          cx={x} 
          cy={y} 
          r={5} 
          fill={color}
          onClick={(e) => handleClick(e)}
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
          fill="white">{index}</text>
      </>
    );
  };

  function toColor(num) {
    let col = 0
    col = tScale(col_dict[num]);
    let c = 0;
    if (num == 'Apus_apus') {
      c = '#123123'
    } else if (num == 'Turdus_merula') {
      c = '#998822'
    }
    return c;
}

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
              height={boundsHeight}
              x={xScale(data.x[dataIndex.current])}
              y={yScale(data.y[dataIndex.current])}
              color={toColor(data.label[dataIndex.current])}
              index={dataIndex.current}
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
