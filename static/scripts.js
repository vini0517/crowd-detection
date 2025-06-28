// Clear all displayed content
function clearContent() {
  // Clear processed image
  let processedImage = document.getElementById("processedImage");
  processedImage.src = "#";
  processedImage.style.display = "none";

  // Clear density map
  let densityImage = document.getElementById("densityImage");
  densityImage.src = "#";
  densityImage.style.display = "none";

  // Clear depth map
  let depthMap = document.getElementById("depthMap");
  depthMap.src = "#";
  depthMap.style.display = "none";

  // Clear status message
  document.getElementById("status").innerText = "";

  // Clear overcrowding message
  let overcrowdedMessage = document.getElementById("overcrowdedMessage");
  overcrowdedMessage.style.display = "none";

  // Clear high-density message
  let highDensityMessage = document.getElementById("highDensityMessage");
  highDensityMessage.style.display = "none";

  // Clear alert box
  let alertBox = document.getElementById("alertBox");
  alertBox.innerText = "";
  alertBox.style.display = "none";
  alertBox.className = ""; // Reset class
}

// Update file label with selected file name and display uploaded image
document.getElementById("imageInput").addEventListener("change", function () {
  let file = this.files[0];
  let fileLabel = document.getElementById("fileLabel");
  fileLabel.innerText = file ? file.name : "Choose File";

  // Clear previous processed content and errors
  clearContent();

  // Display uploaded image immediately in the same row
  if (file) {
    let uploadedImage = document.getElementById("uploadedImage");
    uploadedImage.src = URL.createObjectURL(file);
    uploadedImage.style.display = "block";
  }
});

// Show alert messages
function showAlert(message, type = "error") {
  let alertBox = document.getElementById("alertBox");
  alertBox.innerText = message;
  alertBox.style.display = "block";
  alertBox.className = type; // "error" or "success"
}

// Upload and process image
async function uploadImage() {
  let input = document.getElementById("imageInput");
  let file = input.files[0];

  if (!file) {
    showAlert("Please select an image!", "error");
    return;
  }

  // Validate file type
  let validTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!validTypes.includes(file.type)) {
    showAlert("Invalid file type! Please upload a JPEG or PNG image.", "error");
    return;
  }

  let formData = new FormData();
  formData.append("image", file);

  let status = document.getElementById("status");
  let uploadButton = document.getElementById("uploadButton");
  let loader = document.getElementById("loader");

  status.innerText = "Processing...";
  uploadButton.disabled = true;
  loader.style.display = "block";

  try {
    let response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    let data = await response.json();

    if (data.error) {
      showAlert(data.error, "error");
      status.innerText = "Error processing image!";
      return;
    }

    status.innerText = "Processing complete!";

    // Update crowd count
    let crowdCount = document.getElementById("crowdCount");
    crowdCount.innerText = data.total_count;

    // Show overcrowding message if applicable
    let overcrowdedMessage = document.getElementById("overcrowdedMessage");
    if (data.is_overcrowded) {
      overcrowdedMessage.style.display = "block";
    } else {
      overcrowdedMessage.style.display = "none";
    }

    // Show high-density message if applicable
    let highDensityMessage = document.getElementById("highDensityMessage");
    if (data.is_high_density) {
      highDensityMessage.style.display = "block";
    } else {
      highDensityMessage.style.display = "none";
    }

    // Update processed image
    let processedImage = document.getElementById("processedImage");
    processedImage.src = `/static/processed/${data.processed_image}`;
    processedImage.style.display = "block";

    // Update density map
    let densityImage = document.getElementById("densityImage");
    densityImage.src = `/static/processed/${data.density_map}`;
    densityImage.style.display = "block";

    // Update depth map
    let depthMap = document.getElementById("depthMap");
    depthMap.src = `/static/processed/${data.depth_map}`;
    depthMap.style.display = "block";
  } catch (error) {
    console.error("Error:", error);
    status.innerText = "An error occurred!";
    showAlert("Failed to process the image. Please try again.", "error");
  } finally {
    uploadButton.disabled = false;
    loader.style.display = "none";
  }
}