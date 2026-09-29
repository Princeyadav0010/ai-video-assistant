# 🎬 AI Video Assistant

An AI-powered video and meeting assistant that transforms long-form content into searchable, actionable insights using speech-to-text, Generative AI, and Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://ai-video-assistant-ewm7cjmlckikzhnsj9glcp.streamlit.app/)

**Try the application:**  
https://ai-video-assistant-ewm7cjmlckikzhnsj9glcp.streamlit.app/

---

## 📝 Project Overview

AI Video Assistant is a Streamlit-based application designed to analyze YouTube videos, audio/video files, and PDF documents.

The application can:

- Convert speech into text using OpenAI Whisper
- Support English, Hindi, and Hinglish workflows
- Generate AI-powered meeting/video titles
- Create concise summaries
- Extract action items
- Identify key decisions
- Detect open questions and follow-up topics
- Enable context-aware question answering using RAG
- Provide an interactive chat interface for the analyzed content

The goal is to reduce the time required to understand long videos, meetings, lectures, and documents.

---

## ✨ Features

### 🎙️ Speech Transcription
Uses OpenAI Whisper for local speech-to-text transcription.

### 🌐 Language Support
Supports:

- English → English
- Hindi → English
- English → Hinglish
- Hindi → Hinglish

Hinglish output is generated using Roman/English characters.

### 🤖 Generative AI Analysis
Google Gemini is used for:

- Session title generation
- Summarization
- Action-item extraction
- Decision extraction
- Open-question extraction
- Context-aware responses

### 📋 Meeting Intelligence

Automatically extracts:

- Important tasks
- Responsible owners
- Deadlines when available
- Key decisions
- Unresolved questions
- Important context

### 🔎 RAG-based Q&A

The transcript is converted into embeddings and stored in ChromaDB.

Users can then ask questions about the analyzed content and receive context-aware answers based on the transcript.

### 📄 PDF Analysis

Users can upload text-based PDF documents and analyze their content using the same AI pipeline.

### 💬 Interactive Chat

After processing, users can chat with the analyzed transcript using the RAG pipeline.

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Programming Language
- Python

### Speech Recognition
- OpenAI Whisper

### Generative AI
- Google Gemini API

### AI / NLP
- LangChain
- Hugging Face Sentence Transformers

### Vector Database
- ChromaDB

### Document Processing
- pypdf

### Media Processing
- yt-dlp
- FFmpeg
- Pydub

### Deployment
- Streamlit Community Cloud

---

## 🏗️ Architecture / Workflow

```text
                 ┌─────────────────────┐
                 │   User Input        │
                 │ YouTube / Audio /   │
                 │ Video / PDF         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Content Extraction │
                 │ yt-dlp / pypdf      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Whisper Model     │
                 │ Speech → Text       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Gemini AI         │
                 │ Title / Summary /   │
                 │ Extraction          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Embeddings        │
                 │ Hugging Face        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     ChromaDB        │
                 │   Vector Storage    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    RAG Chat         │
                 │ Context-aware Q&A   │
                 └─────────────────────┘
