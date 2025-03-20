FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 4000

# Command to run the application
CMD ["python", "server/app.py", "--port", "4000"] 