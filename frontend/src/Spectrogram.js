import { useState, useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import axios from "axios";

const width=700
const height=400

export const MakeSpectrogram = ({
  data
})  => {
    const [initSpec, setInitSpec] = useState(null);
    let specData = data;
    const fileSpec = "test.json";
    const canvasRef = useRef(null);

    if (!initSpec) {
      console.log('trying to get initial pic')
      const fetchData = async () => {
        try {
          const response3 = await axios.get(fileSpec);
          setInitSpec(response3.data);
        } catch (error) {
          console.error("Error fetching data:", error);
        }
      };
      fetchData();
    }
    if (!specData && initSpec) {
      specData = initSpec
    }

    useEffect(() => {
      if (specData && canvasRef.current) {
        console.log("inside MakeSpec")
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
    
        const numRows = specData.length;
        const numCols = specData[0].length;
    
        // Create an offscreen canvas to handle image data
        const offscreenCanvas = document.createElement("canvas");
        const offscreenCtx = offscreenCanvas.getContext("2d");
    
        // Set the offscreen canvas to the size of the heatmap
        offscreenCanvas.width = numCols;
        offscreenCanvas.height = numRows;
    
        const imageData = offscreenCtx.createImageData(numCols, numRows);
        const pixels = imageData.data;
    
        // Calculate min and max for normalization
        const minVal = Math.min(...specData.flat());
        const maxVal = Math.max(...specData.flat());
    
        for (let i = 0; i < numRows; i++) {
          for (let j = 0; j < numCols; j++) {
            // const value = specData[i][j];
            const value = specData[numRows-i-1][j];
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
    }, [specData, width, height]);

    return (
        <canvas
          ref={canvasRef}
          style={{ marginLeft: 0, border: "1px solid #ccc" }}
        />
    )
}