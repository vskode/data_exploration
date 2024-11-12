import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import axios from "axios";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const filePath1 = "/files/BermudaHumpbackWhale_20090111_003000_passt_umap.json";
  const filePath2 = "/files/BermudaHumpbackWhale_20090111_003000_passt_umap.json";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response1 = await axios.get(filePath1);
        const response2 = await axios.get(filePath2);
        setEmbeddings({ 'data1': response1.data, 'data2': response2.data });
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
  
  return (
    <div style={{ display: "flex" }}>
      <ScatterPlot
        width={width / 2}
        height={height}
        data={embeddings.data1}
        setSpecData={setSpecData}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#e85252"}
      />
      <ScatterPlot
        width={width / 2}
        height={height}
        data={embeddings.data2}
        setSpecData={setSpecData}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#6689c6"}
      />
      <MakeSpectrogram 
        data={specData}
      />
    </div>
  );
};
