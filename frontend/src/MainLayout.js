import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import axios from "axios";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const filePath1 = "/files/embeddings/2024-11-15_17-26___umap-bird_dawnchorus-birdnet/borneo_sunrise_20240208-063500_birdnet_umap.json";
  const filePath2 = "/files/embeddings/2024-11-15_17-25___umap-bird_dawnchorus-perch/borneo_sunrise_20240208-063500_perch_umap.json";
  const filepaths = [filePath1, filePath2];
  // response = {};

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response_dict = {};
        for (let i = 0; i < filepaths.length; i++) {
          const response = await axios.get(filepaths[i]);
          response_dict[String('data'+(i+1))] =  response.data;
        }
        setEmbeddings(response_dict);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);
  
  if (loading) {
    return <div>Loading...</div>;
  }
  
  const plots = [];
  // for (embedding in embeddings) {
  for (let i = 0; i < Object.keys(embeddings).length; i++) {
    plots.push(
    <ScatterPlot
        width={width / 2}
        height={height}
        data={embeddings[String('data'+(i+1))]}
        setSpecData={setSpecData}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#e85252"}
      />      
    )
  }


  return (
    <div style={{ display: "flex" }}>
      {plots}
      <MakeSpectrogram 
        data={specData}
      />
    </div>
  );
};
