const fields = [
  "crim",
  "zn",
  "indus",
  "chas",
  "nox",
  "rm",
  "age",
  "dis",
  "rad",
  "tax",
  "ptratio",
  "b",
  "lstat",
];

const resultEl = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");

function collectInput() {
  const payload = {};
  for (const key of fields) {
    const value = parseFloat(document.getElementById(key).value);
    if (Number.isNaN(value)) {
      throw new Error(`Invalid value for ${key}`);
    }
    payload[key] = value;
  }
  return payload;
}

async function predictHousePrice() {
  try {
    resultEl.textContent = "Predicting...";
    const payload = collectInput();

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Request failed");
    }

    const data = await response.json();
    resultEl.textContent = `Estimated Price: $${data.predicted_price.toLocaleString()}`;
  } catch (error) {
    resultEl.textContent = `Error: ${error.message}`;
  }
}

predictBtn.addEventListener("click", predictHousePrice);
