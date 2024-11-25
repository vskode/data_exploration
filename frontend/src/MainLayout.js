import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import Dropdown from "react-bootstrap/Dropdown";
import Form from "react-bootstrap/Form";
import Container from "react-bootstrap/Container";
import axios from "axios";
import { CheckboxDropdown } from "./CheckboxDropdown";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [cursorPosition, setCursorPosition] = useState();
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState([
        { id: "em", label: "Exact Mass", checked: true },
      ]);
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
            response_dict[`embeddings${i + 1}`] = {
              'data': response.data,
              'name': dicts[i].split("___")[1].split('/')[0]
            };
          }
          setEmbeddings(response_dict);
          // Directly set new items from response_dict
          const newItems = Object.values(response_dict).map((item) => ({
            id: item.name, // Use `name` as ID
            label: item.name, // Use `name` as label
            checked: false, // Default to unchecked
          }));
          setItems(newItems);
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
  for (let i = 0; i < Object.keys(items).length; i++) {
    if (!items[i].checked) {
      continue;
    }
    plots.push(
      <ScatterPlot
        key={i}
        width={width / 2}
        height={height}
        data={embeddings[`embeddings${i + 1}`]['data']}
        setSpecData={setSpecData}
        cursorPosition={cursorPosition}
        setCursorPosition={setCursorPosition}
        color={"#e85252"}
      />
    );
  }

  return (
    <Container fluid>
      {/* {BasicExample()} */}
      <CheckboxDropdown items={items} setItems={setItems} />
      <div style={{ display: "flex" }}>
        {plots}
        <MakeSpectrogram data={specData} />
      </div>
    </Container>
  );
};
