import { useState, useEffect } from "react";
import { ScatterPlot } from "./ScatterPlot";
import { MakeSpectrogram } from "./Spectrogram";
import Container from "react-bootstrap/Container";
import axios from "axios";
import * as d3 from "d3";
import { CheckboxDropdown } from "./CheckboxDropdown";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [hoveredPlotId, setHoveredPlotId] = useState(null);
  const [globalTimestamp, setGlobalTimestamp] = useState(null);
  
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState([
    { id: "model_name", label: "model-dataset-dimreduction", checked: true },
  ]);
  // const path = { "path": "files/embeddings/CallType_umap/" };
  const path = { "path": "files/embeddings/dcase/dcase/" };

  const enrichDict = (dict) => {
    let lengths = dict[`metadata`][`file_lengths (s)`];
    let dims = dict[`metadata`][`embedding_dimensions`];
    let audiofiles = dict[`metadata`][`audio_files`];
    
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
        last_timestamp += lengths[dim];
    }
    dict['timestamps'] = timestamps;
    dict['time_within_file'] = time_within_file;
    dict['audio_filenames'] = audio_filenames;

    return dict;
  }


  // Fetch dictionaries from the backend
  const getDictionaries = async () => {
    let dicts = [];
    try {
      const url = "http://127.0.0.1:8000/getDictionaries/";
      const response = await axios.post(url, path);
      console.log("Dictionaries received:", response.data.dicts); // Log response
      dicts = response.data.dicts; // Update dictionaries state
      
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
          let model_array = [];
          dicts = await getDictionaries();
          const labels = await axios.get('files/embeddings/dcase/labels.json')
          for (let i = 0; i < dicts.length; i++) {
            console.log("Fetching file from:", dicts[i]); // Log file path being requested
            const embedding = await axios.get(dicts[i]);
            let model_name = embedding.data.metadata.model_name;
            model_array.push(model_name);

            response_dict[model_name] = {
              'data': embedding.data,
              'name': dicts[i].split("___")[1].split('/')[0],
              'model_name': model_name
            };
            response_dict[model_name]['data']['index'] = Array.from(
              {length: response_dict[model_name]['data'].x.length}, 
              (_, n) => n
            );  
            response_dict[model_name].data = enrichDict(response_dict[model_name].data);

            response_dict[model_name]['data'].labels = labels.data[model_name];
            

          }
          setEmbeddings(response_dict);
          // Directly set new items from response_dict
          const newItems = Object.values(response_dict).map((item) => ({
            id: item.model_name, // Use `name` as ID
            label: item.name, // Use `name` as label
            checked: ['birdnet', 'insect66'].includes(item.model_name) ? true : false // Default to checked if 'birdnet'
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

  console.log("Global timestamp:", globalTimestamp);
  const uniqueLabels = [...new Set(embeddings['birdnet'].data.labels.ground_truth)];  // Extract unique labels
  const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(uniqueLabels);

  const plots = [];
  for (let i = 0; i < Object.keys(items).length; i++) {
    const model_name = items[i].id;
    if (!items[i].checked) {
      continue;
    }
    plots.push(
      <ScatterPlot
        plotId={i}
        width={width / 2}
        height={height}
        data={embeddings[model_name]['data']}
        colorScale={colorScale}
        setSpecData={setSpecData}
        globalTimestamp={globalTimestamp}
        setGlobalTimestamp={setGlobalTimestamp}
        hoveredPlotId={hoveredPlotId}
        setHoveredPlotId={setHoveredPlotId}
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
