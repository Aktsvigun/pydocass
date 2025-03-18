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

## Running Locally

**1. Clone Repo**

```bash
git clone https://github.com/your-username/pydocass.git
```

**2. Install Dependencies**

```bash
npm i
```

**3. Set Up Environment Variables**

Copy the example environment file and add your API key:

```bash
cp .env.local.example .env.local
```

**4. Run App**

```bash
npm run dev
```

## Docker Support

The application includes Docker configuration for both development and production environments.

### Development

To start the application in development mode with Docker:

```bash
# Set up your environment variables first
cp .env.local.example .env.local
# Edit .env.local to add your API key

# Start the development container
docker-compose -f docker-compose.dev.yml up --build
```

This configuration:
- Uses node:20-alpine as the base image
- Mounts your local code as a volume for hot reloading
- Runs the application in development mode with `npm run dev`
- Exposes port 3000

### Production

To build and run the application for production:

```bash
# Set up your environment variables
cp .env.local.example .env.local
# Edit .env.local to add your API key

# Build and start the production container
docker-compose up --build
```

This configuration:
- Uses node:20-alpine as the base image
- Builds the application with `npm run build`
- Runs the application in production mode with `npm start`
- Exposes port 3000

## API Documentation

The application uses the following API endpoints:

- `/api/pydocass/document` - Generate documentation for Python code

## Technologies Used

- Next.js
- React
- TypeScript
- Gravity UI
- CodeMirror
- Nebius AI Models
