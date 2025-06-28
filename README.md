📸 Sample Output Screenshots

Below are examples of how the web app displays crowd detection results.

Web Page

![WebPage](https://github.com/user-attachments/assets/352e4e24-8258-4d33-b7c3-d658627134a2)


🔴 Overcrowded Detection Example
This shows an image where multiple individuals were detected by the YOLOv8 model, indicating **High crowd density**.

![Overcrowded Output](https://github.com/user-attachments/assets/380b229f-0d09-48c3-b3d9-a82f00d2ae3a)


✅ Normal / Low Density Detection Example
This shows an image where **No overcrowding** was detected.

![Normal Crowd Output](https://github.com/user-attachments/assets/674bad47-e064-4f18-ac1d-7c10f4190458)


📦 SETUP INSTRUCTIONS FOR CROWD DETECTION PROJECT

1️⃣ Open Anaconda Prompt or Terminal

2️⃣ Navigate to the project directory:

    cd path\to\your\project\folder

Example (Windows):

    cd C:\Users\YourName\Desktop\PROJECT\crowd_detection

3️⃣ Activate your Conda environment (make sure 'crowddetect' already exists):

    conda activate crowddetect

4️⃣ Install required Python libraries:

    pip install -r requirements.txt

5️⃣ Run the Flask web application:

    python app.py

6️⃣ Open your web browser and go to:

    http://127.0.0.1:5000/

💡 Optional:
- Place images to test inside the `uploads/` folder or upload from the web UI.
- Output images will be saved in the `static/` folder (like processed, density, and depth images).
- Make sure `crowddetection.pt` and `model.py` are in the project root for detection to work.

🛠 Tip:
If you haven't created the Conda environment yet:

    conda create -n crowddetect python=3.10
    conda activate crowddetect
