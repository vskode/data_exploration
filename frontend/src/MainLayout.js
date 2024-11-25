import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import axios from "axios";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const path = { "path": "files/embeddings/umap_nn15/" };

  // Fetch dictionaries from the backend
  const getDictionaries = async () => {
    let dicts = [];
    try {
      const url = "http://127.0.0.1:8000/getDictionaries/";
      const response = await axios.post(url, path);
      console.log("Dictionaries received:", response.data.dicts); // Log response
      dicts = response.data.dicts; // Update dictionaries state
      // setDictionariesLoaded(true); // Set the dictionaries as loaded
      return dicts;
    } catch (error) {
      console.error("Error fetching dictionaries:", error);
    }
    return;
  };

  // Fetch embeddings only if dictionaries are populated and loaded
  useEffect(() => {
    const fetchEmbeddings = async () => {
      let dicts = [];
        try {
          const response_dict = {};
          dicts = await getDictionaries();
          for (let i = 0; i < dicts.length; i++) {
            console.log("Fetching file from:", dicts[i]); // Log file path being requested
            const response = await axios.get(dicts[i]);
            response_dict[`data${i + 1}`] = response.data;
          }
          setEmbeddings(response_dict);
          console.log("Embeddings received:", response_dict); // Log response
          // }
        } catch (error) {
          console.error("Error fetching embeddings:", error);
        } finally {
          setLoading(false);
        }
    };

    fetchEmbeddings();
  }, []); // This effect runs when `dictionariesLoaded` or `dictionaries` change

  if (loading) {
    return <div>Loading...</div>;
  }

  const plots = [];
  for (let i = 0; i < Object.keys(embeddings).length; i++) {
    plots.push(
      <ScatterPlot
        key={i}
        width={width / 2}
        height={height}
        data={embeddings[`data${i + 1}`]}
        setSpecData={setSpecData}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#e85252"}
      />
    );
  }

  return (
    <div style={{ display: "flex" }}>
      {plots}
      <MakeSpectrogram data={specData} />
    </div>
  );
};
