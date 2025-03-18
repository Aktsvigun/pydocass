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
