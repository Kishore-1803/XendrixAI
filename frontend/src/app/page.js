"use client";
import { useState, useEffect, useRef } from "react";
import styles from "./page.module.css";
// Import Prism for syntax highlighting
import Prism from "prismjs";
import "prismjs/components/prism-c";
import "prismjs/components/prism-cpp";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-python";
import "prismjs/components/prism-css";
import "prismjs/components/prism-jsx";
import "prismjs/components/prism-java";
import "prismjs/components/prism-typescript";
import "prismjs/components/prism-json";
import "prismjs/components/prism-bash";
import "prismjs/themes/prism-tomorrow.css";
import 'katex/dist/katex.min.css'; // Import KaTeX CSS
import katex from 'katex';
import { BarChart, PieChart, LineChart, ScatterChart } from 'recharts';
import { Bar, Pie, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
// Add more languages as needed

export default function Home() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeChatIndex, setActiveChatIndex] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUploading, setFileUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [supportedLanguages, setSupportedLanguages] = useState(["English"]);
  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [showDocumentList, setShowDocumentList] = useState(false);
  // New state for typing animation
  const [isTyping, setIsTyping] = useState(false);
  const [currentTypingText, setCurrentTypingText] = useState("");
  const [fullResponseText, setFullResponseText] = useState("");
  const [typingSpeed, setTypingSpeed] = useState(30); // ms per character
  const [isGeneratingChart, setIsGeneratingChart] = useState(false);
  const [imagePrompt, setImagePrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);

  
  const chatBoxRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const typingRef = useRef(null);

  // Fetch all chat histories and supported languages on load
  useEffect(() => {
    fetchChatHistory();
    fetchDocuments();
    fetchSupportedLanguages();
  }, []);

  // Auto-scroll to bottom of chat when messages change or during typing
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [chat, currentTypingText]);

  useEffect(() => {
    // Process any LaTeX in the rendered content
    const mathElements = document.querySelectorAll('.katex-render');
    mathElements.forEach(element => {
      try {
        katex.render(element.getAttribute('data-latex'), element, {
          throwOnError: false,
          displayMode: element.getAttribute('data-display') === 'true'
        });
      } catch (error) {
        console.error('KaTeX error:', error);
      }
    });
  }, [chat]); // Re-run when chat messages change

  // Focus input field when chat changes
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [chat]);

  // Apply syntax highlighting after rendering code blocks
  useEffect(() => {
    Prism.highlightAll();
  }, [chat]);

  // Typing animation effect
  useEffect(() => {
    let typingTimeout;
  
    if (isTyping && fullResponseText) {
      if (currentTypingText.length < fullResponseText.length) {
        typingTimeout = setTimeout(() => {
          setCurrentTypingText(fullResponseText.slice(0, currentTypingText.length + 1));
        }, typingSpeed);
      } else {
        // Typing animation is complete
        setIsTyping(false);
        setChat((prev) => {
          const updatedChat = [...prev];
          if (updatedChat.length > 0) {
            updatedChat[updatedChat.length - 1] = {
              ...updatedChat[updatedChat.length - 1],
              text: fullResponseText,
              isAnimated: false,
            };
          }
          return updatedChat;
        });
        setFullResponseText("");
        setCurrentTypingText("");
      }
    }
  
    return () => clearTimeout(typingTimeout);
  }, [isTyping, currentTypingText, fullResponseText, typingSpeed]);


  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const fetchChatHistory = async () => {
    try {
      const res = await fetch("http://localhost:8000/chats");
      const data = await res.json();
      setChatHistory(data.chats || []);
    } catch (err) {
      setError("Failed to fetch chat history. Server might be down.");
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch("http://localhost:8000/documents");
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  const fetchSupportedLanguages = async () => {
    try {
      const res = await fetch("http://localhost:8000/languages");
      const data = await res.json();
      setSupportedLanguages(data.languages || ["English"]);
    } catch (err) {
      console.error("Failed to fetch supported languages:", err);
    }
  };

  const FileUploadUI = () => {
    // Display file size in a readable format
    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
  
    return (
      <div className={styles.fileUploadContainer}>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf,.docx,.csv,.txt"
          className={styles.fileInput}
          id="fileInput"
        />
        
        {!selectedFile ? (
          <label htmlFor="fileInput" className={styles.fileInputButton}>
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
              className={styles.fileIcon}
            >
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
            <span>Attach File</span>
          </label>
        ) : (
          <div className={styles.selectedFileContainer}>
            <div className={styles.selectedFileInfo}>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                className={styles.fileTypeIcon}
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
              <div className={styles.fileDetails}>
                <span className={styles.fileName}>{selectedFile.name}</span>
                <span className={styles.fileSize}>{formatFileSize(selectedFile.size)}</span>
              </div>
              <button 
                onClick={clearSelectedFile}
                className={styles.removeFileButton}
                aria-label="Remove file"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  width="16" 
                  height="16" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            {fileUploading && (
              <div className={styles.uploadProgress}>
                <div className={styles.progressBar}></div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };
  
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
        setSelectedFile(file);
        // Add a visual indication that file is selected
        const fileName = file.name.length > 20 
            ? file.name.substring(0, 17) + '...' 
            : file.name;
        setInput(prev => prev + `\n[File attached: ${fileName}]`);

        // Automatically upload the file after selection
        const uploadResult = await uploadFileOnly();
        if (uploadResult) {
            console.log("File uploaded successfully:", uploadResult);
        } else {
            console.error("File upload failed.");
        }
    }
  };

  const clearSelectedFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    // Remove the file indication from the input
    setInput(input.replace(/\n\[File attached:.*?\]/, ""));
  };

  const uploadFileOnly = async () => {
    if (!selectedFile) return null;
    
    setFileUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await fetch("http://localhost:8000/upload_file", {
        method: "POST",
        body: formData,
      });
      
      const result = await response.json();
      setFileUploading(false);
      
      // Add the new document to the documents list
      fetchDocuments();
      
      return result;
    } catch (err) {
      setError("Failed to upload file. Please try again.");
      setFileUploading(false);
      return null;
    }
  };

  const handleDocumentSelect = (document) => {
    setSelectedDocument(document);
    setShowDocumentList(false);
    
    // Create a new chat with the document
    createChatWithDocument(document.id);
  };

  const createChatWithDocument = async (documentId) => {
    try {
      const res = await fetch(`http://localhost:8000/new_chat?document_id=${documentId}&language=${selectedLanguage}`, {
        method: "POST"
      });
      const result = await res.json();
      
      await fetchChatHistory();
      
      // Find the index of the newly created chat and load it
      const newChatIndex = chatHistory.findIndex(chat => chat.id === result.chat_id);
      if (newChatIndex !== -1) {
        loadChat(newChatIndex);
      } else {
        // If the chat isn't found in the history yet, refresh and try again
        await fetchChatHistory();
        const updatedIndex = chatHistory.findIndex(chat => chat.id === result.chat_id);
        if (updatedIndex !== -1) {
          loadChat(updatedIndex);
        }
      }
    } catch (err) {
      setError("Failed to create a new chat with document. Please try again.");
    }
  };

  // Function to detect if a message might be requesting a chart/visualization
  const isChartRequest = (text) => {
    const chartKeywords = [
      'chart', 'graph', 'plot', 'visualize', 'visualization', 'bar chart',
      'pie chart', 'line graph', 'scatter plot', 'histogram', 'visualisation',
      'data visualization', 'display data', 'show data', 'analytics', 
      'visual representation', 'dashboard', 'statistics'
    ];
    
    const lowerText = text.toLowerCase();
    return chartKeywords.some(keyword => lowerText.includes(keyword));
  };

  const handleVisualize = async (message) => {
    if (!message.trim()) return;

    // Add a user message indicating visualization request
    const userMsg = {
      sender: "You",
      text: `Visualize: ${message}`,
      language: selectedLanguage
    };
    setChat((prev) => [...prev, userMsg]);
    setIsGeneratingChart(true);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      // Determine the chart type from the message
      let chartType = "bar"; // default
      const lowerMessage = message.toLowerCase();
      if (lowerMessage.includes("pie")) chartType = "pie";
      else if (lowerMessage.includes("line")) chartType = "line";
      else if (lowerMessage.includes("scatter")) chartType = "scatter";

      // Prepare request data
      const requestData = {
        query: message,
        chart_type: chartType,
        language: selectedLanguage
      };
      
      // Add chat_id if available
      if (activeChatIndex !== null && chatHistory[activeChatIndex]) {
        requestData.chat_id = chatHistory[activeChatIndex].id;
      }
      
      // Add document_id if available
      if (selectedDocument && chatHistory[activeChatIndex]?.document_id) {
        requestData.document_id = chatHistory[activeChatIndex].document_id;
      }

      // Send request to visualization endpoint
      const response = await fetch("http://localhost:8000/visualize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error");
      }

      const result = await response.json();

      // Check if visualization data and image path are returned
      if (result && result.visualization_data) {
        // Create a visualization message with all necessary data
        const visualizationMsg = {
          sender: "AI",
          text: `Here's a ${chartType} chart visualization based on your request.`,
          language: selectedLanguage,
          isVisualization: true,
          chartData: result.visualization_data,
          image_path: result.image_path // Store the image path
        };
        
        // Add to chat
        setChat((prev) => [...prev, visualizationMsg]);
      } else {
        throw new Error("Failed to generate visualization");
      }

      // Refresh chat history to keep everything in sync
      await fetchChatHistory();
      
    } catch (err) {
      console.error("Error in handleVisualize:", err);
      setError(`Error: ${err.message || "Failed to generate visualization."}`);
      
      const errorMsg = {
        sender: "AI",
        text: `Sorry, I encountered an error while generating the visualization: ${err.message || "Failed to process your request"}`,
        language: selectedLanguage
      };
      setChat((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setIsGeneratingChart(false);
    }
  };

  const handleGenerateImage = async (prompt) => {
    if (!prompt.trim()) return;

    // Add a user message indicating image generation request
    const userMsg = {
      sender: "You",
      text: `Generate Image: ${prompt}`,
      language: selectedLanguage
    };
    setChat((prev) => [...prev, userMsg]);
    setIsGeneratingChart(true); // Reuse this state for loading indicator
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/generate_image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error");
      }

      const result = await response.json();

      if (result && result.image_base64) {
        // Create a message with the generated image
        const imageMsg = {
          sender: "AI",
          text: `Here's an image generated based on your prompt.`,
          language: selectedLanguage,
          image_path: result.image_base64 // base64 string as inline image
        };

        setChat((prev) => [...prev, imageMsg]);
      } else {
        throw new Error("Failed to generate image");
      }

    } catch (err) {
      console.error("Error in handleGenerateImage:", err);
      setError(`Error: ${err.message || "Failed to generate image."}`);

      const errorMsg = {
        sender: "AI",
        text: `Sorry, I encountered an error while generating the image: ${err.message || "Unknown issue"}`,
        language: selectedLanguage
      };
      setChat((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setIsGeneratingChart(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() && !selectedFile) return; // Ensure at least a message or file is provided

    const userMsg = {
      sender: "You",
      text: selectedFile
        ? `${input || "Please analyze this file"}\n[File: ${selectedFile.name}]`
        : input,
      language: selectedLanguage
    };
    setChat((prev) => [...prev, userMsg]);
    
    // Store the message for chart detection
    const messageText = input.trim() || "Please analyze this file";
    const isChartReq = isChartRequest(messageText);
    
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("message", messageText);
      formData.append("language", selectedLanguage);
      
      // If we have a selected document, use its ID
      if (selectedDocument && chatHistory[activeChatIndex]?.document_id) {
        formData.append("document_id", chatHistory[activeChatIndex].document_id);
      }
      
      // If we have a selected file, attach it
      if (selectedFile) {
        formData.append("file", selectedFile);
      }
      
      // If we have an active chat, use its ID
      if (activeChatIndex !== null && chatHistory[activeChatIndex]) {
        formData.append("chat_id", chatHistory[activeChatIndex].id);
      } 

      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error");
      }

      const result = await response.json();

      // Add AI message only if we have a valid response
      if (result && result.response) {
        // Start the typing animation
        const aiResponseText = result.response;
        
        // Set the animation speed based on content length and chart detection
        // Chart responses may have more data, so we speed up the typing
        const baseSpeed = isChartReq ? 10 : 30;
        setTypingSpeed(Math.max(5, Math.min(30, baseSpeed - (aiResponseText.length / 1000))));
        
        // Add an initial empty message that will be animated
        const aiMsg = { 
          sender: "AI", 
          text: "",
          language: selectedLanguage,
          isAnimated: true
        };
        setChat((prev) => [...prev, aiMsg]);
        
        // Set the full text and start typing animation
        setFullResponseText(aiResponseText);
        setCurrentTypingText("");
        setIsTyping(true);
      } else {
        throw new Error("Received empty response from server");
      }

      // Clear the selected file after sending
      if (selectedFile) {
        clearSelectedFile();
      }

      // Refresh sidebar to update the chat name dynamically
      await fetchChatHistory();
      
      // If we don't have an active chat yet, set it based on the result
      if (activeChatIndex === null) {
        const newChatIndex = chatHistory.findIndex(chat => chat.id === result.chat_id);
        if (newChatIndex !== -1) {
          setActiveChatIndex(newChatIndex);
        } else {
          await fetchChatHistory();
          const updatedIndex = chatHistory.findIndex(chat => chat.id === result.chat_id);
          if (updatedIndex !== -1) {
            setActiveChatIndex(updatedIndex);
          }
        }
      }
    } catch (err) {
      console.error("Error in sendMessage:", err);
      setError(`Error: ${err.message || "Failed to get response from the AI. Please try again later."}`);

      // Add an error message to the chat for visibility
      const errorMsg = {
        sender: "AI",
        text: `Sorry, I encountered an error: ${err.message || "Failed to process your request"}`,
        language: selectedLanguage
      };
      setChat((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = async () => {
    try {
      const response = await fetch(`http://localhost:8000/new_chat?language=${selectedLanguage}`, { 
        method: "POST" 
      });
      const result = await response.json();
      
      setChat([]);
      setSelectedDocument(null);
      setActiveChatIndex(null);
      await fetchChatHistory();
      
      if (inputRef.current) {
        inputRef.current.focus();
      }
    } catch (err) {
      setError("Failed to create a new chat. Please try again later.");
    }
  };

  const loadChat = async (index) => {
    try {
      // Get the chat ID
      const chatId = chatHistory[index].id;
      
      // Fetch the full chat data with messages
      const response = await fetch(`http://localhost:8000/chats/${chatId}`);
      if (!response.ok) {
        throw new Error("Failed to load chat data");
      }
      
      const chatData = await response.json();
      
      // Process each message to identify and parse visualization data
      const processedMessages = chatData.messages.map(msg => {
        // Check if this message contains visualization data
        if (msg.sender === "AI" && typeof msg.text === 'string' && msg.text.includes('visualization_data:')) {
          try {
            // Extract the visualization data
            const visualizationMatch = msg.text.match(/visualization_data:(\{.*?\})/s);
            if (visualizationMatch && visualizationMatch[1]) {
              const visualizationData = JSON.parse(visualizationMatch[1]);
              
              // Return message with parsed chart data
              return {
                ...msg,
                isVisualization: true,
                chartData: visualizationData
              };
            }
          } catch (error) {
            console.error("Error parsing visualization data:", error);
          }
        }
        
        return msg;
      });
      
      // Set the processed messages to chat state
      setChat(processedMessages);
      setActiveChatIndex(index);
      
      // Set language based on chat's language
      if (chatData.language) {
        setSelectedLanguage(chatData.language);
      }
      
      // If the chat has a document, set it as the selected document
      if (chatData.document_id) {
        const doc = documents.find(d => d.id === chatData.document_id);
        if (doc) {
          setSelectedDocument(doc);
        }
      } else {
        setSelectedDocument(null);
      }
      
      // Close sidebar on mobile after selecting a chat
      if (window.innerWidth < 768) {
        setShowSidebar(false);
      }
    } catch (error) {
      console.error("Error loading chat:", error);
      setError("Failed to load chat. Please try again.");
    }
  };

  const deleteChat = async (index) => {
    try {
      const chatId = chatHistory[index].id;
      const res = await fetch(`http://localhost:8000/chats/${chatId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        // After deletion, refresh the chat history
        await fetchChatHistory();
        // If the active chat was deleted, clear the chat area
        if (index === activeChatIndex) {
          setChat([]);
          setActiveChatIndex(null);
          setSelectedDocument(null);
        }
      } else {
        throw new Error("Failed to delete chat.");
      }
    } catch (err) {
      setError("Failed to delete the chat. Please try again later.");
    }
  };
  
  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);
  };

  const getChatPreview = (chatItem) => {
    if (!chatItem.messages || chatItem.messages.length === 0) return "Empty chat";
    const firstUserMsg = chatItem.messages.find(msg => msg.sender === "You");
    if (firstUserMsg) {
      return firstUserMsg.text.length > 25 
        ? `${firstUserMsg.text.slice(0, 25)}...` 
        : firstUserMsg.text;
    }
    return "New conversation";
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleDateString();
  };

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar);
  };

  const toggleDocumentList = () => {
    setShowDocumentList(!showDocumentList);
  };

  const processFormattedText = (text) => {
    if (!text) return '';
    
    // Temporarily replace LaTeX content to avoid processing it
    const mathExpressions = [];
    
    // Replace block LaTeX ($$...$$)
    text = text.replace(/\$\$(.*?)\$\$/g, (match, latex) => {
      const placeholder = `__MATH_BLOCK_${mathExpressions.length}__`;
      mathExpressions.push({
        placeholder,
        latex,
        display: true
      });
      return placeholder;
    });
    
    // Replace inline LaTeX ($...$)
    text = text.replace(/\$(.*?)\$/g, (match, latex) => {
      const placeholder = `__MATH_INLINE_${mathExpressions.length}__`;
      mathExpressions.push({
        placeholder,
        latex,
        display: false
      });
      return placeholder;
    });
    
    // Process bold text (**text**)
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Process italic text (*text*) but prevent overlapping with bold
    // This regex avoids matching inside already-processed bold tags
    text = text.replace(/(?<!\\)\*(?!\*)((?:\\.|[^\*])*?)(?<!\\)\*(?!\*)/g, '<em>$1</em>');
    
    // Process underscores for alternative bold/italic
    text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
    text = text.replace(/_([^_]*)_/g, '<em>$1</em>');
    
    // Replace newlines with break tags
    text = text.replace(/\n/g, '<br>');
    
    // Put back the LaTeX expressions
    mathExpressions.forEach(({placeholder, latex, display}) => {
      const katexPlaceholder = display ? 
        `<div class="katex-render" data-latex="${escapeHtml(latex)}" data-display="true"></div>` :
        `<span class="katex-render" data-latex="${escapeHtml(latex)}" data-display="false"></span>`;
      
      text = text.replace(placeholder, katexPlaceholder);
    });
    
    return text;
  };
  const escapeHtml = (unsafe) => {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };
  
  // Update the formatMessage function to better handle chart data
  const formatMessage = (message, isAnimated = false) => {
    // If this is the currently animating message, use the current typing text
    const textToFormat = isAnimated ? currentTypingText : message.text;

    // Check for image_path in the message object (for loaded chats)
    if (message.image_path) {
      // Use the correct backend endpoint for images
      const imgSrc = `http://localhost:8000/images/${message.image_path}`;
      const downloadFile = async () => {
        try {
          const response = await fetch(imgSrc);
          if (!response.ok) throw new Error("Network response was not ok");
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = message.image_path;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
        } catch (err) {
          alert("Failed to download image.");
        }
      };
      return (
        <>
          <span dangerouslySetInnerHTML={{ __html: processFormattedText(textToFormat) }}></span>
          <div className={styles.chartContainer}>
            <img src={imgSrc} alt="Generated Chart" className={styles.generatedImage} />
            <button className={styles.downloadButton} onClick={downloadFile}>
              Download Image
            </button>
          </div>
        </>
      );
    }

     // Handle visualization images specially
    if (message.isVisualization && message.chartData) {
      // Extract chart data
      const chartData = message.chartData;
      const chartType = chartData.chart_type;
      
      // Create formatted data for Recharts
      const formattedData = chartData.data.map(point => ({
        name: point.label,
        value: point.value
      }));
      
      return (
        <>
          <p>{textToFormat.split('![Chart]')[0]}</p>
          <div className={styles.chartContainer}>
            <ResponsiveContainer width="100%" height={300}>
              {chartType === 'bar' && (
                <BarChart data={formattedData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              )}
              {chartType === 'pie' && (
                <PieChart>
                  <Pie 
                    data={formattedData} 
                    dataKey="value" 
                    nameKey="name" 
                    cx="50%" 
                    cy="50%" 
                    outerRadius={100} 
                    fill="#8884d8" 
                    label
                  >
                    {formattedData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={`#${Math.floor(Math.random()*16777215).toString(16)}`} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              )}
              {chartType === 'line' && (
                <LineChart data={formattedData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" />
                </LineChart>
              )}
              {chartType === 'scatter' && (
                <ScatterChart>
                  <CartesianGrid />
                  <XAxis dataKey="name" type="category" />
                  <YAxis dataKey="value" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Values" data={formattedData} fill="#8884d8" />
                </ScatterChart>
              )}
            </ResponsiveContainer>
          </div>
        </>
      );
    }
    const base64ImageRegex = /!\[Chart\]\((data:image\/[^;]+;base64,[^\)]+)\)/;
    const base64Match = textToFormat.match(base64ImageRegex);
    
    if (base64Match) {
      const imageData = base64Match[1]; // This contains the full base64 data URL
      const textBeforeImage = textToFormat.split('![Chart]')[0];
      const downloadBase64 = () => {
        const link = document.createElement('a');
        link.href = imageData;
        link.download = 'chart.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      };
      
      return (
        <>
          {textBeforeImage && <span dangerouslySetInnerHTML={{ __html: processFormattedText(textBeforeImage) }}></span>}
          <div className={styles.chartContainer}>
            <img src={imageData} alt="Generated Chart" className={styles.generatedImage} />
            <button className={styles.downloadButton} onClick={downloadBase64}>
              Download Image
            </button>
          </div>
        </>
      );
    }

    if (typeof message.text === 'string' && message.text.includes('visualization_data:')) {
    try {
      // Extract the visualization data if it's embedded in JSON format
      const visualizationMatch = message.text.match(/visualization_data:(\{.*?\})/s);
      if (visualizationMatch && visualizationMatch[1]) {
        const parsedData = JSON.parse(visualizationMatch[1]);
        const chartType = parsedData.chart_type || 'bar';
        
        const formattedData = parsedData.data.map(point => ({
          name: point.label,
          value: point.value
        }));
        
        const textBeforeVisualization = message.text.split('visualization_data:')[0];
        
        return (
          <>
            <span dangerouslySetInnerHTML={{ __html: processFormattedText(textBeforeVisualization) }}></span>
            <div className={styles.chartContainer}>
              <ResponsiveContainer width="100%" height={300}>
                {chartType === 'bar' && (
                  <BarChart data={formattedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#8884d8" />
                  </BarChart>
                )}
                {chartType === 'pie' && (
                  <PieChart>
                    <Pie 
                      data={formattedData} 
                      dataKey="value" 
                      nameKey="name" 
                      cx="50%" 
                      cy="50%" 
                      outerRadius={100} 
                      fill="#8884d8" 
                      label
                    >
                      {formattedData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={`#${Math.floor(Math.random()*16777215).toString(16)}`} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                )}
                {chartType === 'line' && (
                  <LineChart data={formattedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#8884d8" />
                  </LineChart>
                )}
                {chartType === 'scatter' && (
                  <ScatterChart>
                    <CartesianGrid />
                    <XAxis dataKey="name" type="category" />
                    <YAxis dataKey="value" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Values" data={formattedData} fill="#8884d8" />
                  </ScatterChart>
                )}
              </ResponsiveContainer>
            </div>
          </>
        );
      }
    } catch (error) {
      console.error("Error parsing visualization data:", error);
    }
  }
    // If chart data exists, extract it and remove from text
    let processedText = textToFormat;
    
    // First, split the text by code blocks
    const parts = processedText.split(/(```[\s\S]*?```)/g);
    
    return (
      <>
        {parts.map((part, index) => {
          if (part.startsWith("```") && part.endsWith("```")) {
            // Handle code blocks (unchanged)
            let code = part.slice(3, -3);
            let language = "javascript"; // default language
            
            // Check if there's a language specified
            const firstLineBreak = code.indexOf('\n');
            if (firstLineBreak > 0) {
              const possibleLang = code.substring(0, firstLineBreak).trim();
              // Common language identifiers
              const validLangs = ["javascript", "java", "python", "css", "html", "jsx", "typescript", "cpp", "json", "go", "ruby", "php", "sh", "sql"];
              
              if (validLangs.includes(possibleLang)) {
                language = possibleLang;
                code = code.substring(firstLineBreak + 1);
              }
            }
            
            // Map short language names to their full names for Prism
            const langMap = {
              js: "javascript", 
              py: "python",
              ts: "typescript",
            };
            
            const prismLanguage = langMap[language] || language;

            const copyToClipboard = () => {
              navigator.clipboard.writeText(code).then(() => {
                alert("Code copied to clipboard!");
              }).catch(() => {
                alert("Failed to copy code.");
              });
            };

            return (
              <div key={index} className={styles.codeCanvas}>
                <div className={styles.codeHeader}>
                  <span>{language}</span>
                  <button onClick={copyToClipboard} className={styles.copyButton}>
                    Copy
                  </button>
                </div>
                <pre className={`language-${prismLanguage}`}>
                  <code className={`language-${prismLanguage}`}>
                    {code}
                  </code>
                </pre>
              </div>
            );
          } else {
            // Check for file attachment notation
            if (part.includes("[File:") || part.includes("[File attached:")) {
              // Your existing file attachment handling code...
              const fileRegex = /\[File(?:\s+attached)?:\s*(.*?)\]/g;
              const matches = [...part.matchAll(fileRegex)];
              
              // Get unique filenames
              const uniqueFiles = [...new Set(matches.map(match => match[1].trim()))];
              
              const textBeforeFile = part.split(/\[File(?:\s+attached)?:/)[0];
              
              // Function to get file icon and color based on extension
              const getFileInfo = (filename) => {
                const extension = filename.split('.').pop().toLowerCase();
                
                // File icon mapping
                const iconMap = {
                  'pdf': {
                    icon: (
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                      </svg>
                    ),
                    color: '#E74C3C'
                  },
                  // Other file types
                };
                
                // Default file icon
                return iconMap[extension] || {
                  icon: (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                      <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                  ),
                  color: '#7F8C8D'
                };
              };
              
              return (
                <span key={index}>
                  {/* Process text before file attachment for formatting */}
                  {textBeforeFile && <span dangerouslySetInnerHTML={{ __html: processFormattedText(textBeforeFile) }}></span>}
                  {uniqueFiles.length > 0 && (
                    <div className={styles.fileAttachmentContainer}>
                      {uniqueFiles.map((filename, i) => {
                        const fileInfo = getFileInfo(filename);
                        return (
                          <div key={i} className={styles.fileAttachmentBadge} style={{ borderLeft: `4px solid ${fileInfo.color}` }}>
                            <div className={styles.fileIcon} style={{ color: fileInfo.color }}>
                              {fileInfo.icon}
                            </div>
                            <span>{filename}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </span>
              );
            }
            
            // Process normal text with formatting
            return isAnimated ? 
              <span key={index} className={styles.typingAnimation} dangerouslySetInnerHTML={{ __html: processFormattedText(part) }}></span> : 
              <span key={index} dangerouslySetInnerHTML={{ __html: processFormattedText(part) }}></span>;
          }
        })}
      </>
    );
  };
  
  // Xendrix logo component
  const XendrixLogo = () => (
    <div className={styles.logoContainer}>
      <img 
        src="/logo.png" // Ensure the path is correct relative to the public folder
        alt="Xendrix Logo" 
        className={styles.logo}
      />
      <span className={styles.logoText}>Xendrix</span>
    </div>
  );

  return (
    <main className={styles.container}>
      <div className={`${styles.sidebar} ${showSidebar ? styles.sidebarOpen : styles.sidebarClosed}`}>
        <div className={styles.sidebarHeader}>
          {/* Replace text logo with actual logo */}
          <XendrixLogo />
          <button onClick={startNewChat} className={styles.newChatBtn}>
            <span className={styles.btnIcon}>+</span> New Chat
          </button>
        </div>
        
        <div className={styles.languageSelector}>
          <label htmlFor="language-select">Language:</label>
          <select 
            id="language-select" 
            value={selectedLanguage}
            onChange={handleLanguageChange}
            className={styles.languageSelect}
          >
            {supportedLanguages.map(lang => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>
        </div>
        
        <div className={styles.historyContainer}>
          <h3>Chat History</h3>
          {chatHistory.length === 0 && (
            <div className={styles.noChats}>No previous chats</div>
          )}
          <ul className={styles.chatList}>
            {chatHistory.map((chatItem, index) => (
              <li
                key={chatItem.id || index}
                onClick={() => loadChat(index)}
                className={
                  index === activeChatIndex
                    ? styles.chatItemActive
                    : styles.chatItem
                }
              >
                <div className={styles.chatItemContent}>
                  <span className={styles.chatItemTitle}>
                    {chatItem.name || getChatPreview(chatItem)}
                  </span>
                  <div className={styles.chatItemMeta}>
                    {chatItem.language && (
                      <span className={styles.chatItemLanguage}>
                        {chatItem.language}
                      </span>
                    )}
                    {chatItem.document_id && (
                      <span className={styles.chatItemDocumentBadge}>
                        📄
                      </span>
                    )}
                    <span className={styles.chatItemDate}>
                      {formatTimestamp(chatItem.timestamp)}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(index);
                  }}
                  className={styles.deleteButton}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </div>
        
        {/* Add sidebar toggle button inside the sidebar */}
        <button 
          className={styles.hamburgerMenu}
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          style={{ position: 'absolute', top: '10px', right: '-40px', background: '#252836', color: '#fff' }}
        >
          {showSidebar ? "◀" : "▶"}
        </button>
      </div>

      <div className={`${styles.chatContainer} ${!showSidebar ? styles.chatContainerFull : ''}`}>
        <header className={styles.header}>
          <h1>
            <img 
              src="/logo.png" // Ensure the path is correct relative to the public folder
              alt="Xendrix Logo" 
              style={{ marginRight: '8px', width: '100px', height: '100px' }}
            />
          </h1>
          <div className={styles.headerRight}>
            {selectedDocument && (
              <div className={styles.activeChatInfo}>
                <span className={styles.activeChatDocument}>
                  📄 {selectedDocument.name}
                </span>
              </div>
            )}
            <div className={styles.headerLanguage}>
              {selectedLanguage}
            </div>
          </div>
        </header>

        <div className={styles.chatBox} ref={chatBoxRef}>
          {chat.length === 0 && (
            <div className={styles.welcomeMessage}>
              <h2>Welcome to Xendrix Chat</h2>
              <p>Ask anything to get started!</p>
              <p>You can also upload PDF, DOCX, or CSV files for analysis.</p>
              <p>Select a language from the dropdown to chat in different languages.</p>
              <p>Use the "Show Documents" button to access previously uploaded documents.</p>
            </div>
          )}
          {chat.map((msg, i) => (
            <div
              key={i}
              className={`${
                msg.sender === "You"
                  ? styles.userMessage
                  : styles.aiMessage
              } ${msg.isAnimated ? styles.animateResponse : ''}`}
            >
                <div className={styles.messageBubble}>
                  <div className={styles.messageHeader}>
                    {msg.language && msg.language !== "English" && (
                      <span className={styles.messageLanguage}>{msg.language}</span>
                    )}
                  </div>
                <div className={styles.messageContent}>
                  {formatMessage(msg, msg.isAnimated)}
                </div>
              </div>
            </div>
          ))}
          {(loading || isGeneratingChart) && (
            <div className={styles.loadingContainer}>
              <div className={styles.typingIndicator}>
                <img src="/logo.png" alt="Xendrix Logo" className={styles.typingLogo} />
                <div className={styles.typingDots}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className={styles.typingText}>
                  {isGeneratingChart ? "Generating visualization..." : "Thinking..."}
                </span>
              </div>
            </div>
          )}
          {error && <div className={styles.errorMessage}>{error}</div>}
        </div>

        <div className={styles.inputContainer}>
          <FileUploadUI />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            rows={3}
            className={styles.textarea}
            placeholder={`Type your message in ${selectedLanguage}... (Press Enter to send)`}
          />
          <div className={styles.buttonGroup}>
            <button 
              onClick={() => handleVisualize(input)}
              className={`${styles.visualizeButton} ${!input.trim() ? styles.disabled : ''}`}
              disabled={loading || !input.trim()}
              title="Generate visualization from your data"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path>
                <rect x="16" y="3" width="4" height="4"></rect>
                <rect x="3" y="16" width="4" height="4"></rect>
                <rect x="9" y="9" width="4" height="4"></rect>
                <line x1="5" y1="20" x2="19" y2="6"></line>
              </svg>
            </button>
            <button 
              onClick={() => handleGenerateImage(input)}
              className={`${styles.generateImageButton} ${!input.trim() ? styles.disabled : ''}`}
              disabled={loading || !input.trim()}
              title="Generate image based on your prompt"
            >
              🖼️
            </button>
                      <button 
              onClick={sendMessage} 
              className={styles.sendButton}
              disabled={loading || (!input.trim() && !selectedFile)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}