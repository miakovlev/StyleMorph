# Style Transfer Application

This project is a Streamlit-based demo that applies neural style transfer to user-supplied images.
It focuses on a clean, minimal UI so the visual transformation is the main focus.
The app loads pretrained models and lets you experiment with different artistic styles.
It is meant as a portfolio-friendly, "before/after" showcase with quick results.
The codebase includes a simple inference pipeline, a Streamlit front end, and Docker support.

## Highlights
- Interactive image upload and stylization in the browser.
- Pretrained style transfer models for fast inference.
- Dockerized setup for consistent local runs.

## Before / After Gallery (placeholders)
> **Insert images here**: Please replace the placeholders below with real screenshots.

### Example 1
**Before**
![Before placeholder - replace with screenshot](docs/images/before-1.png)

**After**
![After placeholder - replace with screenshot](docs/images/after-1.png)

### Example 2
**Before**
![Before placeholder - replace with screenshot](docs/images/before-2.png)

**After**
![After placeholder - replace with screenshot](docs/images/after-2.png)

## Building and Running the Docker Container

To build and run the Docker container, follow these steps:
1. Build the Docker image: `docker build -t stylusfusion .`
2. Run the Docker container: `docker run -p 8501:8501 stylusfusion`

This will start the Streamlit application on port 8501. You can access it by navigating to http://localhost:8501 in your web browser.

## Working Application
You can access the working application at the following link:
https://stylusfusion-98ytu.ondigitalocean.app/
