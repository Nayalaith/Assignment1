# Movie Recommendation System

A Transformer-based movie recommendation system using BERT embeddings to suggest movies across the five major streaming platforms. Built with **FastAPI** for API endpoints and **Streamlit** for frontend deployment, the system uses deep learning and transformer-based techniques to provide personalized movie recommendations.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Technical Details](#technical-details)

## Project Overview

This recommendation system utilizes **BERT embeddings** to compute semantic similarity between movie metadata and user preferences, creating a recommendation list tailored to users' interests. The model is trained on datasets from five major streaming platforms, ensuring a wide variety of content.

Key technologies:
- **BERT Transformer** for movie embeddings
- **FastAPI** for backend API
- **Streamlit** for frontend UI
- **Docker** for easy deployment

## Directory Structure

The project is organized as follows:

```plaintext
.
├── code
│   ├── datasets
│   │   └── *.csv                # Datasets from major streaming platforms
│   ├── deployment
│   │   ├── api                  # FastAPI code
│   │   ├── app                  # Streamlit app code
│   │   └── Dockerfile.yml       # Docker setup for both API and app
│   └── model
│       └── Recommendation_system.py # The main recommendation system python code 
|── Embeddings ── *.h5                 # Trained BERT embeddings for recommendation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

```
### Key Files

- **datasets**: Contains `.csv` files for each platform's movie dataset.
- **deployment**: Contains the `api` and `app` directories, along with `Dockerfile.yml` for deployment.
- **models**: Stores `.h5` files with BERT embeddings, not directly used in the app but serve as the core of the recommendation process.
- **main.py**: Entry point to run the application.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nayalaith/Movies-Recommendation-System.git
   cd Movies-Recommendation-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you are using Docker, ensure you have Docker and Docker Compose installed on your machine.
4. Build the Docker image (optional):
   ```bash
   docker-compose -f deployment/Dockerfile.yml up --build
   ```
5. Run the FastAPI server (if not using Docker):
   ```bash
    uvicorn deployment.api.main:app --reload
   ```
6. Run the Streamlit frontend (if not using Docker):
7. ```bash
   streamlit run deployment/app/main.py
   ```
## Technical Details
- BERT Embeddings: The recommendation engine uses pre-trained BERT embeddings saved in .h5 format in the models directory.
- FastAPI: Acts as a bridge between the embedding model and the user inputs, efficiently handling API requests.
- Streamlit: Provides a clean and interactive UI for users to receive movie recommendations based on their preferences.
