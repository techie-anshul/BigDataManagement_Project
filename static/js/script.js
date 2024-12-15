document.getElementById("predictButton").addEventListener("click", function () {
    const symbol = document.getElementById("symbol").value.trim();

    if (!symbol) {
        document.getElementById("results").textContent = "Please enter a valid stock symbol.";
        return;
    }

    fetch(`/predict/${symbol}`)
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById("results");
            if (data.error) {
                resultsDiv.textContent = `Error: ${data.error}`;
            } else {
                resultsDiv.innerHTML = `
                    <p><strong>Model saved at:</strong> ${data.message}</p>
                    <p><strong>Last 5 Predictions:</strong> ${data.predictions.slice(-5).join(', ')}</p>
                    <p><strong>Actual:</strong> ${data.actual.slice(-5).join(', ')}</p>
                `;
            }
        })
        .catch(error => {
            console.error("Error fetching prediction:", error);
            document.getElementById("results").textContent = "An error occurred while fetching prediction.";
        });
});

