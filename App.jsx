
import { useEffect, useRef, useState } from "react";
import * as faceapi from "@vladmandic/face-api";
import "./App.css";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

// Adjust these values if your particular glasses PNG needs fine tuning.
const GLASSES_WIDTH_FACTOR = 2.35;
const GLASSES_VERTICAL_OFFSET = 0.02;

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const frameRef = useRef(null);
  const runningRef = useRef(false);
  const modeRef = useRef("glasses");

  const [mode, setMode] = useState("glasses");
  const [modelsReady, setModelsReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Loading face detection models...");
  const [fileName, setFileName] = useState("");

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      try {
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

        if (!cancelled) {
          setModelsReady(true);
          setStatus("Models ready. Start your camera.");
        }
      } catch (error) {
        console.error("Model loading failed:", error);
        if (!cancelled) {
          setStatus("Model loading failed. Check your internet connection.");
        }
      }
    }

    loadModels();

    return () => {
      cancelled = true;
      stopCamera();
    };
  }, []);

  async function startCamera() {
    try {
      if (!modelsReady) {
        setStatus("Please wait for models to load.");
        return;
      }

      setStatus("Starting camera...");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });

      await video.play();

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      runningRef.current = true;
      setCameraOn(true);
      setStatus("Camera active. Upload your accessory PNG.");

      detectLoop();
    } catch (error) {
      console.error("Camera error:", error);
      setStatus(
        error.name === "NotAllowedError"
          ? "Camera permission denied. Allow camera access in browser settings."
          : "Could not start camera. Check camera connection and permissions."
      );
    }
  }

  function stopCamera() {
    runningRef.current = false;

    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    setCameraOn(false);
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("Please select a valid image.");
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      if (overlayRef.current?.src?.startsWith("blob:")) {
        URL.revokeObjectURL(overlayRef.current.src);
      }

      overlayRef.current = image;
      setFileName(file.name);
      setStatus("Accessory uploaded. Look straight at the camera.");
    };

    image.onerror = () => {
      URL.revokeObjectURL(imageUrl);
      setStatus("Image could not be loaded. Try another PNG.");
    };

    image.src = imageUrl;
    event.target.value = "";
  }

  function getEyeCenter(points) {
    return {
      x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
      y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
    };
  }

  function drawGlasses(landmarks, ctx) {
    const image = overlayRef.current;
    if (!image) return;

    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    if (!leftEye.length || !rightEye.length) return;

    // Calculate the centre of both eyes.
    const leftCenter = getEyeCenter(leftEye);
    const rightCenter = getEyeCenter(rightEye);

    // Distance and angle between eyes.
    const dx = rightCenter.x - leftCenter.x;
    const dy = rightCenter.y - leftCenter.y;
    const eyeDistance = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx);

    // Centre of the glasses.
    const centerX = (leftCenter.x + rightCenter.x) / 2;
    const centerY = (leftCenter.y + rightCenter.y) / 2;

    // Scale glasses according to the user's face.
    const glassesWidth = eyeDistance * GLASSES_WIDTH_FACTOR;
    const imageRatio = image.naturalHeight / image.naturalWidth;
    const glassesHeight = glassesWidth * imageRatio;

    ctx.save();
    ctx.translate(centerX, centerY + eyeDistance * GLASSES_VERTICAL_OFFSET);
    ctx.rotate(angle);

    ctx.drawImage(
      image,
      -glassesWidth / 2,
      -glassesHeight / 2,
      glassesWidth,
      glassesHeight
    );

    ctx.restore();
  }

  function drawNecklace(landmarks, ctx) {
    const image = overlayRef.current;
    if (!image) return;

    const jaw = landmarks.getJawOutline();
    const chin = jaw[8];
    const leftJaw = jaw[0];
    const rightJaw = jaw[16];

    const faceWidth = rightJaw.x - leftJaw.x;
    const centerX = (leftJaw.x + rightJaw.x) / 2;

    const width = faceWidth * 1.65;
    const height = width * (image.naturalHeight / image.naturalWidth);

    ctx.drawImage(
      image,
      centerX - width / 2,
      chin.y + faceWidth * 0.12,
      width,
      height
    );
  }

  async function detectLoop() {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) {
      frameRef.current = requestAnimationFrame(detectLoop);
      return;
    }

    try {
      const result = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: 0.4,
          })
        )
        .withFaceLandmarks();

      if (!runningRef.current) return;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (result && overlayRef.current) {
        if (modeRef.current === "glasses") {
          drawGlasses(result.landmarks, ctx);
        } else {
          drawNecklace(result.landmarks, ctx);
        }
      }
    } catch (error) {
      console.error("Detection error:", error);
    }

    if (runningRef.current) {
      frameRef.current = requestAnimationFrame(detectLoop);
    }
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">G</span>
          GenFit
        </div>
        <span className="topbar-tag">AI VIRTUAL TRY-ON</span>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">YOUR VIRTUAL STYLING STUDIO</p>
          <h1>
            Try it on.
            <br />
            <span>Own your look.</span>
          </h1>
          <p className="subtitle">
            Upload your favourite accessories and see how they look on you.
          </p>
        </div>

        <section className="studio">
          <div className="studio-heading">
            <div>
              <p className="eyebrow">LIVE PREVIEW</p>
              <h2>Your virtual mirror</h2>
            </div>
            <span className={`live-indicator ${cameraOn ? "active" : ""}`}>
              <span className="live-dot" />
              {cameraOn ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="camera-video"
            />

            <canvas
              ref={canvasRef}
              className="camera-canvas"
            />

            {!cameraOn && (
              <div className="camera-placeholder">
                <div className="camera-placeholder-icon">◎</div>
                <h3>Your preview appears here</h3>
                <p>Start your camera to try on accessories.</p>
              </div>
            )}
          </div>

          <p className="status-message">{status}</p>

          <div className="controls">
            {!cameraOn ? (
              <button
                className="primary-button"
                onClick={startCamera}
                disabled={!modelsReady}
              >
                Start Camera
              </button>
            ) : (
              <button className="secondary-button" onClick={stopCamera}>
                Stop Camera
              </button>
            )}

            <label className="upload-button">
              + Upload PNG
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleUpload}
              />
            </label>
          </div>

          <div className="mode-controls">
            <button
              className={mode === "glasses" ? "mode-button selected" : "mode-button"}
              onClick={() => setMode("glasses")}
            >
              👓 Glasses Mode
            </button>

            <button
              className={mode === "necklace" ? "mode-button selected" : "mode-button"}
              onClick={() => setMode("necklace")}
            >
              💎 Necklace Mode
            </button>
          </div>

          {fileName && <p className="file-name">Uploaded: {fileName}</p>}
        </section>
      </section>

      <footer className="footer">
        GenFit · AI-powered virtual accessory try-on
      </footer>
    </main>
  );
}
