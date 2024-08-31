import { useState, useEffect } from "react";
import { LineChart } from "./LineChart";
import axios from "axios";

export const LineChartSyncCursor = ({ width = 700, height = 400 }) => {
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true); // Track loading state
  const filePath1 = "/files/umap_embeds/2024-08-31_15-29___umap-humpbacks-aves/BermudaHumpbackWhale_20090111_003000_aves_umap.json"
  const filePath2 = "/files/umap_embeds/2024-08-31_15-27___umap-humpbacks-birdnet/BermudaHumpbackWhale_20090111_003000_birdnet_umap.json"
  // Fetch data on component mount


  useEffect(() => {
    const fetchData = async () => {
      try {
        // const response = await axios.get('/data.json');
        const response1 = await axios.get(filePath1);
        const response2 = await axios.get(filePath2);
        // npyArray = await n.load(response);
        // console.log(response1.data);
        setEmbeddings({'data1': response1.data, 'data2': response2.data});
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false); // Set loading to false once data is fetched or an error occurs
      }
    };
    
    fetchData();
  }, []); // Empty dependency array means this effect runs once on mount
  // console.log(embeddings);

  // Conditional rendering based on loading state
  if (loading) {
    return <div>Loading...</div>; // You can replace this with any loading indicator or spinner
  }
  else {
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
      </div>
    )
  }
};