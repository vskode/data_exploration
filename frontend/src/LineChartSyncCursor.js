import { useState, useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import { LineChart } from "./LineChart";
import { ShowSpectogram } from "./ShowSpectogram";
import axios from "axios";

export const LineChartSyncCursor = ({ width = 700, height = 400 }) => {
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [spec, setSpec] = useState(null);
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef(null);
  const filePath1 = "/files/umap_embeds/2024-08-31_15-29___umap-humpbacks-aves/BermudaHumpbackWhale_20090111_003000_aves_umap.json";
  const filePath2 = "/files/umap_embeds/2024-08-31_15-27___umap-humpbacks-birdnet/BermudaHumpbackWhale_20090111_003000_birdnet_umap.json";
  const fileSpec = "test.json";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response1 = await axios.get(filePath1);
        const response2 = await axios.get(filePath2);
        const response3 = await axios.get(fileSpec);

        setEmbeddings({ 'data1': response1.data, 'data2': response2.data });
        setSpec(response3.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
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
    return <div>Loading...</div>;
  }
  
  return (
    <div style={{ display: "flex" }}>
      <LineChart
        width={width / 2}
        height={height}
        data={embeddings.data1}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#e85252"}
      />
      <LineChart
        width={width / 2}
        height={height}
        data={embeddings.data2}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#6689c6"}
      />
      <ShowSpectogram
        width={width / 2}
        height={height}
      />
    </div>
  );
};
