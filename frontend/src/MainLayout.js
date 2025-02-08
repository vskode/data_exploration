import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import Container from "react-bootstrap/Container";
import axios from "axios";
import { CheckboxDropdown } from "./CheckboxDropdown";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [hoveredPlotId, setHoveredPlotId] = useState(null);
  const [globalTimestamp, setGlobalTimestamp] = useState(null);
  
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState([
    { id: "em", label: "Exact Mass", checked: true },
  ]);
  // const path = { "path": "files/embeddings/CallType_umap/" };
  const path = { "path": "files/embeddings/xc_umap/" };
  const repeatByCounts = (arr, counts) => 
    arr.flatMap((item, index) => Array(counts[index]).fill(item));
  
  const enrichDict = (response_dict, i) => {
    let lengths = response_dict[`embeddings${i + 1}`][`data`][`metadata`][`file_lengths (s)`];
    let dims = response_dict[`embeddings${i + 1}`][`data`][`metadata`][`embedding_dimensions`];
    let audiofiles = response_dict[`embeddings${i + 1}`][`data`][`metadata`][`audio_files`];
    
    let timestamps = [];
    let time_within_file = [];
    let audio_filenames = [];
    let last_timestamp = 0;
    

    for (let dim = 0; dim < lengths.length; dim++) {
        let step = lengths[dim] / dims[dim][0]; // Step size based on embedding dimension
        let current = 0;
    
        while (current <= lengths[dim]) {
            timestamps.push(current + last_timestamp);
            audio_filenames.push(audiofiles[dim]);
            time_within_file.push(current);
            current += step;
        }
        last_timestamp = lengths[dim];
    }
    response_dict[`embeddings${i + 1}`]['data']['timestamps'] = timestamps;
    response_dict[`embeddings${i + 1}`]['data']['time_within_file'] = time_within_file;
    response_dict[`embeddings${i + 1}`]['data']['audio_filenames'] = audio_filenames;

    return response_dict;
  }


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
          let response_dict = {};
          dicts = await getDictionaries();
          for (let i = 0; i < dicts.length; i++) {
            console.log("Fetching file from:", dicts[i]); // Log file path being requested
            const response = await axios.get(dicts[i]);
            response_dict[`embeddings${i + 1}`] = {
              'data': response.data,
              'name': dicts[i].split("___")[1].split('/')[0]
            };
            response_dict[`embeddings${i + 1}`]['data']['index'] = Array.from(
              {length: response_dict[`embeddings${i + 1}`]['data'].label.length}, 
              (_, n) => n
            );  
            const repeat_array = response_dict[`embeddings${i + 1}`]['data']
                                .metadata['embedding_dimensions']
                                .map((i) => i[0]);

            response_dict = enrichDict(response_dict, i);

            // console.log(timestamps);
            
            const label_array = response_dict[`embeddings${i + 1}`]['data'].label;
            // Example:
            const new_labels = repeatByCounts(label_array, repeat_array);
            
            response_dict[`embeddings${i + 1}`]['data'].label = new_labels;
            

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
        plotId={i}
        width={width / 2}
        height={height}
        data={embeddings[`embeddings${i + 1}`]['data']}
        setSpecData={setSpecData}
        globalTimestamp={globalTimestamp}
        setGlobalTimestamp={setGlobalTimestamp}
        hoveredPlotId={hoveredPlotId}
        setHoveredPlotId={setHoveredPlotId}
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
