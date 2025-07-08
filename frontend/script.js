async function classifyImage() {
  const imageUrl = document.getElementById("imageUrl").value.trim();
  const resultsDiv = document.getElementById("results");
  const predictionsList = document.getElementById("predictions");

  if (!imageUrl) {
    alert("Please enter an image URL.");
    return;
  }

  predictionsList.innerHTML = "Loading...";
  resultsDiv.classList.remove("hidden");

  try {
    const response = await fetch("http://127.0.0.1:8000/api/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ image_url: imageUrl })
    });

    const data = await response.json();

    if (data.predictions) {
      predictionsList.innerHTML = "";
      data.predictions.forEach(pred => {
        const li = document.createElement("li");
        li.textContent = `${pred.label} — ${(pred.score * 100).toFixed(2)}%`;
        predictionsList.appendChild(li);
      });
    } else {
      predictionsList.innerHTML = "Error in prediction.";
    }
  } catch (err) {
    console.error(err);
    predictionsList.innerHTML = "Failed to connect to backend.";
  }
}
