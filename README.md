# Hand Gesture Recognition using OpenCV and MediaPipe

This project uses [MediaPipe](https://github.com/google-ai-edge/mediapipe) and [OpenCV](https://opencv.org/) to recognize hand gestures from a webcam feed in real time — no machine learning training required!

## 📸 Demo

![Gesture Recognition Demo](screenshot.png)

## 🖐️ Recognized Gestures

- Fist
- Open Palm
- Thumbs Up
- Peace Sign ✌️
- OK Sign 👌
- Call Me 🤙

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

Install the required packages using:

```bash
pip install -r requirements.txt
```
📂 Dataset: HaGRID
We evaluated the rule-based hand gesture recognition method using the HaGRID (HAnd Gesture Recognition Image Dataset). This dataset consists of millions of hand gesture images captured in the wild under various lighting, backgrounds, and hand orientations. It provides a more realistic benchmark for testing gesture recognition models compared to webcam input alone.

The evaluation script and results are included in the notebook:
📄 rule_based_methods.ipynb

📉 Conclusion:
The rule-based method performs well in controlled environments such as live webcam feeds, where lighting is stable and gestures are clearly posed. However, when applied to the HaGRID dataset, the method failed is most of the gestures.

