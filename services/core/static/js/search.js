async function runSearch() {
  const query = document.getElementById("query").value.trim();
  const model = document.getElementById("model").value;
  const encodeUrl = MODEL_URLS[model];

  if (!query) {
    alert("Please enter a query.");
    return;
  }

  try {
    // 1. Encode query into embedding
    const encodeResp = await fetch(encodeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ "query": query })
    });

    const encodeData = await encodeResp.json();
    const embedding = encodeData.embedding;
    console.log("Encoded embedding:", embedding);

    // 2. Post embedding to search endpoint
    const searchResp = await fetch(SEARCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ "embedding": embedding, "type": model, "k": 40 })
    });

    const results = await searchResp.json();
    console.log("Search results:", results);

    // 3. Display results
    displayResults(results);
  } catch (err) {
    console.error("Error:", err);
    alert("Something went wrong. Check console.");
  }
}

function displayResults(results) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (results.length === 0) {
    resultsDiv.innerHTML = "<p>No results found.</p>";
    return;
  }

  results.forEach(res => {
    const frameId = res.frame_id;
    const score = res.score;

    const img = document.createElement("img");
    img.src = `${FRAME_IMAGE_PATH}${frameId}.png`;
    img.alt = `Score: ${score}`;
    img.title = `Score: ${score}`;
    img.className = "result-img";

    resultsDiv.appendChild(img);
  });
}
