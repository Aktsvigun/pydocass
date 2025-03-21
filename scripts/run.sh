#!/bin/bash

# Run the backend service
start_backend() {
  echo "Starting backend service..."
  cd backend
  pip install -r requirements.txt
  cd src
  python -m pydocass &
  BACKEND_PID=$!
  cd ../..
  echo "Backend started with PID: $BACKEND_PID"
}

# Run the frontend service
start_frontend() {
  echo "Starting frontend service..."
  cd frontend
  npm install
  npm run dev &
  FRONTEND_PID=$!
  cd ..
  echo "Frontend started with PID: $FRONTEND_PID"
}

# Stop all services
stop_services() {
  echo "Stopping services..."
  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID
    echo "Frontend stopped"
  fi
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID
    echo "Backend stopped"
  fi
  exit 0
}

# Trap SIGINT and SIGTERM
trap stop_services SIGINT SIGTERM

# Start services
start_backend
start_frontend

echo "Services started. Press Ctrl+C to stop all services."

# Wait for user input
wait 