# PyDocAss - Python Documentation Assistant

This repository contains a web application that helps developers document Python code with the assistance of AI.

![PyDoc Assistant](./frontend/public/screenshot.png)

## Project Structure

The project has been organized into a more maintainable structure:

- `frontend/`: Contains the Next.js frontend application
- `backend/`: Contains the Python backend application
- `docker/`: Docker configuration files
- `scripts/`: Utility scripts
- `docs/`: Project documentation

For more details on the project structure, please see [docs/README.md](docs/README.md).

## Running Locally

**1. Frontend**

```bash
cd frontend
npm install
npm run dev
```

**2. Backend**

```bash
cd backend
pip install -r requirements.txt
python -m pydocass
```

**3. Docker**

```bash
docker-compose up -d
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
