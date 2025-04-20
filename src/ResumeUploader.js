import { useState } from "react";
import axios from "axios";

export default function ResumeUploader() {
  const [file, setFile] = useState(null);
  const [predictedRole, setPredictedRole] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a resume file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPredictedRole(response.data.predicted_role);
    } catch (error) {
      console.error("Error uploading file", error);
      alert("Failed to predict job role.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-5">
      <div className="bg-white p-6 rounded-xl shadow-md w-96">
        <h2 className="text-xl font-semibold mb-4">Upload Your Resume</h2>
        <input type="file" accept="application/pdf" onChange={handleFileChange} className="mb-4" />
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
        {predictedRole && (
          <p className="mt-4 text-lg font-medium">Predicted Job Role: <span className="text-green-600">{predictedRole}</span></p>
        )}
      </div>
    </div>
  );
}
