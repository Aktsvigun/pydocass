# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables directly in the image
ENV NEBIUS_API_KEY="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwODM2MzQ1NTA5ODQ4NjQ1NzQwOCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTg5NTg0MzA4NCwidXVpZCI6IjM0ZDFlYTNlLWNkZjAtNDUwZS1hOWQzLTE3MzgxMzkzNTA3ZiIsIm5hbWUiOiJkZXZsaWthIiwiZXhwaXJlc19hdCI6IjIwMzAtMDEtMjhUMTU6MDQ6NDQrMDAwMCJ9.ZKxrXac-vercfiF08dEGIpF4aEm3xYAUfVjBS37U_7M"
ENV DB_CONNECTION="postgresql://nebo:Voblakah)0@public-rw.postgresql-e00jbzyk5f4brjrbxr.backbone-e00ys3jaqren9ke219.msp.eu-north1.nebius.cloud:5432/python-documentation?sslmode=verify-full"
ENV PYTHONUNBUFFERED=1
ENV PORT=4000

# Copy the entire application code
COPY . .

# Create a shell script to print environment variables and start the app
RUN echo '#!/bin/bash\n\
echo "Environment variables:"\n\
echo "NEBIUS_API_KEY is set: ${NEBIUS_API_KEY:0:10}..."\n\
echo "DB_CONNECTION is set: ${DB_CONNECTION:0:20}..."\n\
echo "PORT: $PORT"\n\
\n\
# Start the application\n\
exec python server/app.py --port "${PORT:-4000}"\n\
' > /app/start.sh && chmod +x /app/start.sh

# Install the package in development mode (which also installs dependencies)
RUN pip install -e .

# Expose the port the app will run on
EXPOSE 4000

# Start the application
CMD ["/app/start.sh"]
