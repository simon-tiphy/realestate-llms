

// async function classifyImage() {
//   const imageUrl = document.getElementById("imageUrl").value.trim();
//   const resultsDiv = document.getElementById("results");
//   const predictionsList = document.getElementById("predictions");

//   if (!imageUrl) {
//     alert("Please enter an image URL.");
//     return;
//   }

//   predictionsList.innerHTML = "Loading...";
//   resultsDiv.classList.remove("hidden");

//   try {
//     const response = await fetch("http://127.0.0.1:8000/api/classify", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ image_url: imageUrl })
//     });

//     const data = await response.json();

//     if (data.predictions) {
//       predictionsList.innerHTML = "";
//       data.predictions.forEach(pred => {
//         const li = document.createElement("li");
//         li.textContent = `${pred.label} — ${(pred.score * 100).toFixed(2)}%`;
//         predictionsList.appendChild(li);
//       });
//     } else {
//       predictionsList.innerHTML = "Error in prediction.";
//     }
//   } catch (err) {
//     console.error(err);
//     predictionsList.innerHTML = "Failed to connect to backend.";
//   }
// }

// async function generateText() {
//   const prompt = document.getElementById("prompt").value.trim();
//   const outputDiv = document.getElementById("textOutput");
//   const generatedText = document.getElementById("generatedText");

//   if (!prompt) {
//     alert("Please enter a question.");
//     return;
//   }

//   generatedText.textContent = "Generating...";
//   outputDiv.classList.remove("hidden");

//   try {
//     const response = await fetch("http://127.0.0.1:8000/api/generate", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ prompt })
//     });

//     const data = await response.json();

//     if (data.generated_text) {
//       generatedText.textContent = data.generated_text;
//     } else {
//       generatedText.textContent = "Error generating text.";
//     }
//   } catch (err) {
//     console.error(err);
//     generatedText.textContent = "Failed to connect to backend.";
//   }
// }


const API_BASE = "http://127.0.0.1:8000";

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
    const res = await fetch(`${API_BASE}/api/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl })
    });
    const data = await res.json();

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

async function generateText() {
  const prompt = document.getElementById("prompt").value.trim();
  const outputDiv = document.getElementById("textOutput");
  const generatedPre = document.getElementById("generatedText");

  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }

  generatedPre.textContent = "Generating...";
  outputDiv.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();

    if (data.generated_text) {
      generatedPre.textContent = data.generated_text;
    } else {
      generatedPre.textContent = "Error generating text.";
    }
  } catch (err) {
    console.error(err);
    generatedPre.textContent = "Failed to connect to backend.";
  }
}
