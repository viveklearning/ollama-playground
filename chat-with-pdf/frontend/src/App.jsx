import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const handleFileUpload = async () => {
    if (!file) return alert("Please select a file!");
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        await axios.post("http://127.0.0.1:8000/upload/", formData);
        alert("File uploaded successfully!");
    } catch (error) {
        console.error("Error uploading file:", error);
        alert("Failed to upload file.");
    }
};

const handleAskQuestion = async () => {
    if (!question) return;
    
    const formData = new FormData();
    formData.append("question", question);

    try {
        const response = await axios.post("http://127.0.0.1:8000/ask/", formData);
        setMessages([...messages, { role: "user", text: question }, { role: "bot", text: response.data.answer }]);
        setQuestion(""); 
    } catch (error) {
        console.error("Error asking question:", error);
        alert("Failed to get an answer.");
    }
};

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <h2>PDF Chatbot</h2>
      
      {/* File Upload */}
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleFileUpload}>Upload PDF</button>

      {/* Chat Interface */}
      <div style={{ border: "1px solid #ccc", padding: "10px", marginTop: "10px", height: "300px", overflowY: "auto" }}>
        {messages.map((msg, index) => (
          <p key={index} style={{ color: msg.role === "user" ? "blue" : "green" }}>
            <b>{msg.role === "user" ? "You: " : "Bot: "}</b>{msg.text}
          </p>
        ))}
      </div>

      {/* Question Input */}
      <input 
        type="text" 
        value={question} 
        onChange={(e) => setQuestion(e.target.value)} 
        placeholder="Ask something..." 
        style={{ width: "80%", marginRight: "10px" }} 
      />
      <button onClick={handleAskQuestion}>Ask</button>
    </div>
  );
}

export default App;
