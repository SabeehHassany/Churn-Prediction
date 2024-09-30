# Churn Prediction Flask App

A Flask application that predicts customer churn based on user input. The application is built using Python and machine learning models to make predictions about customer churn probability. The app accepts customer details via a web form and returns a prediction along with the probability of churn.

## Table of Contents

- [Project Overview](#project-overview)
- [Languages and Frameworks](#languages-and-frameworks)
- [Features](#features)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Run the Application](#run-the-application)
- [Run with Docker](#run-with-docker)

## Project Overview

This project uses a machine learning model to predict whether a customer is likely to churn based on several customer features such as `tenure`, `monthly charges`, `phone service`, etc. The model is integrated with a Flask web application to provide an interactive interface where users can input customer information and receive a churn prediction.

## Languages and Frameworks

- **Programming Language**: Python
- **Framework**: Flask (Backend)
- **Machine Learning**: Scikit-learn
- **Containerization**: Docker
- **Libraries**:
  - Pandas
  - NumPy
  - Joblib (for model persistence)
  - Scikit-learn (for the machine learning model)

## Features

- Web-based interface to enter customer details.
- Predict customer churn using a pre-trained machine learning model.
- Display prediction result with probability.
- Containerized using Docker for easy deployment.

## Requirements

Make sure you have the following installed:

- Python 3.9 or later
- Flask
- Docker (if you plan to run the app inside a container)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```
### 2. Create a Virtual Environment

To avoid dependency conflicts, it's recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

### 3. Install Required Packages

Install the necessary Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Run the Application

After setting up the environment and installing dependencies, you can run the Flask app:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

## Build the Docker Image

To build the Docker image for the Flask app, run:

```bash
docker build -t churn-prediction-app .
```
## Run with Docker

Run the container mapping the local machine’s port `5000` to the container’s port `5000`:

```bash
docker run -p 5000:5000 churn-prediction-app
```
