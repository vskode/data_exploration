import { useState, useEffect } from "react";
import { LineChart } from "./LineChart";
import axios from "axios";

export const LineChartSyncCursor = ({ width = 700, height = 400 }) => {
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState();
  const [loading, setLoading] = useState(true); // Track loading state

  // Fetch data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/data.json');
        // console.log(response.data);
        setEmbeddings(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false); // Set loading to false once data is fetched or an error occurs
      }
    };

    fetchData();
  }, []); // Empty dependency array means this effect runs once on mount

  // Conditional rendering based on loading state
  if (loading) {
    return <div>Loading...</div>; // You can replace this with any loading indicator or spinner
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
    </div>
  );
};