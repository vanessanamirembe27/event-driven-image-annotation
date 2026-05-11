Overview

This is an event driven visual oject retrieval architecture system where a user can upload and search images. The system detects objects in each image, stores annotations in a document database, indexes embeddings in a vector store and returns the top most similar images for any search query. The focus of this project is system design, not AI model training.

Architecture

The system is built around five independent services that should not talk directly to each ther but through a message broker(Redis). Each service publishes events when its finished its work and subscribes to events when it needs to be triggered.

Services

Upload Service - accepts an image from CLI.
Inference Service - simulates object detection.
Annotation Service - stores annotation documents for each image. Each document contans the full list of detected objects as nested JSON.
Embedding Service - Indexes embedding vectors so images can be found by similarity. Also handles search queries by running FAISS.
CLI Service - this is the entry point for users to upload and search.

Messages Definition

image.submitted - Fired when a new image is uploaded
inference.completed - Fired when object detection is done.
annotation.stored - fired when annotation is written to the document database.
embedding created - Image is now fully indexed and searchaeable.
annotaion.corrected - fired when a user corrects a mislabeled annotation.
query submitted - fired when a userr submits a search query.
query completed - fired when FAISS search completes.

Topic summary

Topic                 |    Publisher          |    Subscriber(s)
image.submitted       |   Upload Service      |    Inference Service
inference.completed   |   Inference Service   |    Annotation Service
annotation.stored     |   Annotation Service  |    Embedding service
embedding.created     |   Embedding Service   |  
annotation.corrected  |    CLI Service        |    Annotation service, Embedding service
query.submitted       |    CLI service        |    Embedding service
query.completed       |    Embedding service  |    CLI service

Below is the link to the video requested.(uploaded on google drive: https://drive.google.com/file/d/1qaOBqVkHjvEavnDaLCervg6pN7iLdsiJ/view?usp=drive_link
