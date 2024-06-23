#!/bin/bash
# startup.sh

# Wait for the FastAPI server to start
echo "Waiting for FastAPI server to start..."
while ! nc -z localhost 80; do   
  sleep 0.1 # wait for 1/10 of the second before check again
done
echo "FastAPI server started."

# Send the predefined JSON payload to the FastAPI endpoint
curl -X 'POST' \
  'http://localhost/download-videos/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
        "https://www.youtube.com/watch?v=WeF4wpw7w9k&t=6s",
        "https://www.youtube.com/watch?v=2NFwY15tRtA",
        "https://www.youtube.com/watch?v=5dRramZVu2Q"
      ]'