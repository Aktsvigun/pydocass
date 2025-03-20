# PyDocAss - Python Documentation Assistant

A Next.js application that helps you automatically generate documentation for your Python code using Nebius AI models.

## Features

- Generate docstrings, comments, and type annotations for Python code
- Multiple AI model options with varying capabilities and speeds
- Customizable documentation options
- Modern UI with light/dark theme support

## Project Structure

```
pydocass/
├── app/                      # App router components
│   └── api/                  # API routes using app router
│       └── pydocass/         # Python backend service
├── components/               # Shared React components
│   ├── common/               # General purpose UI components
│   ├── forms/                # Form-related components
│   └── layout/               # Layout components
├── config/                   # Configuration files
├── lib/                      # Shared utility functions
├── pages/                    # Pages router
│   └── api/                  # API routes using pages router
├── public/                   # Static assets
├── styles/                   # Global styles
└── types/                    # TypeScript type definitions
```

## Architecture

The application consists of two main components:

1. **Frontend**: A Next.js web application that provides the user interface
2. **Backend**: A Python Flask API that processes code documentation requests using AI models

## Running Locally (Without Docker)

**1. Clone Repo**

```bash
git clone https://github.com/your-username/pydocass.git
```

**2. Install Dependencies**

For the frontend:
```bash
npm i
```

For the backend:
```bash
cd app/api/pydocass
pip install -e .
pip install -r requirements.txt
```

**3. Set Up Environment Variables**

Copy the example environment file and add your API key:

```bash
cp .env.local.example .env.local
```

**4. Configure API Endpoint for Local Development**

For local development without Docker, you need to rename the API endpoint file:

```bash
cd pages/api/pydocass
mv document.ts document.docker.ts
mv document.local.ts document.ts
```

This ensures the frontend uses localhost to connect to your locally running backend.

**5. Run Both Services**

Frontend:
```bash
npm run dev
```

Backend:
```bash
cd app/api/pydocass
python server/app.py
```

The frontend will be available at http://localhost:3000 and the backend API at http://localhost:4000.

## Docker Support

The application includes Docker configuration for both development and production environments. Both configurations run the frontend and backend services together.

### Development

To start the application in development mode with Docker:

```bash
# Set up your environment variables first
cp .env.local.example .env.local
# Edit .env.local to add your API key

# Start the development containers
docker-compose -f docker-compose.dev.yml up --build
```

This configuration:
- Runs both frontend and backend services
- Uses volume mounts for hot reloading of both services
- Exposes frontend on port 3000 and backend on port 4000

### Production

To build and run the application for production:

```bash
# Set up your environment variables
cp .env.local.example .env.local
# Edit .env.local to add your API key

# Build and start the production containers
docker-compose up --build
```

This configuration:
- Runs optimized builds of both frontend and backend
- Configures proper service dependencies
- Exposes frontend on port 3000 and backend on port 4000

## API Documentation

The application uses the following API endpoints:

- Frontend API: `/api/pydocass/document` - Proxy endpoint that forwards requests to the backend
- Backend API: `http://localhost:4000/document` - Processes and generates documentation for Python code

## Technologies Used

- Frontend:
  - Next.js
  - React
  - TypeScript
  - Gravity UI
  - CodeMirror

- Backend:
  - Python
  - Flask
  - OpenAI/Nebius AI Models
