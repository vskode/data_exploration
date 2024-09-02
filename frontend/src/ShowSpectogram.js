import { useState, useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import axios from "axios";

const width=700
const height=400

export const ShowSpectogram = ({
    data=null
})  => {
    const [spec, setSpec] = useState(null);
    const fileSpec = "test.json";
    const [loading, setLoading] = useState(true);
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!data) {
        const fetchData = async () => {
          try {
            const response3 = await axios.get(fileSpec);
            setSpec(response3.data);
          } catch (error) {
            console.error("Error fetching data:", error);
          } finally {
            setLoading(false);
          }
        };
    
        fetchData();
        } else {
            setSpec(data)
        }
      }, []);
    
      useEffect(() => {
        if (spec && canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
      
          const numRows = spec.length;
          const numCols = spec[0].length;
      
          // Create an offscreen canvas to handle image data
          const offscreenCanvas = document.createElement("canvas");
          const offscreenCtx = offscreenCanvas.getContext("2d");
      
          // Set the offscreen canvas to the size of the heatmap
          offscreenCanvas.width = numCols;
          offscreenCanvas.height = numRows;
      
          const imageData = offscreenCtx.createImageData(numCols, numRows);
          const pixels = imageData.data;
      
          // Calculate min and max for normalization
          const minVal = Math.min(...spec.flat());
          const maxVal = Math.max(...spec.flat());
      
          for (let i = 0; i < numRows; i++) {
            for (let j = 0; j < numCols; j++) {
              const value = spec[i][j];
              const normalizedValue = (value - minVal) / (maxVal - minVal);
              const color = d3.interpolateViridis(normalizedValue);
      
              const rgbColor = d3.rgb(color);
              const r = rgbColor.r;
              const g = rgbColor.g;
              const b = rgbColor.b;
      
              const index = (i * numCols + j) * 4;
              pixels[index] = r;         // Red
              pixels[index + 1] = g;     // Green
              pixels[index + 2] = b;     // Blue
              pixels[index + 3] = 255;   // Alpha
            }
          }
      
          offscreenCtx.putImageData(imageData, 0, 0);
      
          // Set canvas size to match LineChart dimensions
          canvas.width = width / 2;
          canvas.height = height;
      
          // Calculate scale to fit the heatmap to the canvas
          const scaleX = canvas.width / numCols;
          const scaleY = canvas.height / numRows;
      
          // Use drawImage to scale the heatmap data
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(offscreenCanvas, 0, 0, numCols, numRows, 0, 0, canvas.width, canvas.height);
        }
      }, [spec, width, height]);
      
      // useMemo((loading) => {
        // if (spec) {
        //   drawHeatmap(spec);
        // }
      // }, [spec]);
      
      if (loading) {
        return <div>LoadingSpec...</div>;
      }

    return (
        <canvas
        ref={canvasRef}
        style={{ marginLeft: 20, border: "1px solid #ccc" }}
        />
    )
}