document.getElementById("predictButton").addEventListener("click", function () {
    const symbol = this.getAttribute("data-symbol");

    fetch(`/predict/${symbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("predictedPrice").textContent = `Error: ${data.error}`;
            } else {
                document.getElementById("predictedPrice").textContent = `Predicted Price: ${data.latest_prediction}`;
            }
        })
        .catch(error => {
            console.error("Error fetching prediction:", error);
            document.getElementById("predictedPrice").textContent = "Error fetching prediction.";
        });
});

