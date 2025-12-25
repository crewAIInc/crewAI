#!/bin/bash

# QRI Trading Dashboard - Start Script
# Starts both backend and frontend in tmux

set -e

PROJECT_DIR="/Users/charafchnioune/Desktop/Code/tools/crewAI247"
SESSION_NAME="crewai"

echo "🚀 Starting QRI Trading Dashboard..."

# Kill existing session if exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Kill any processes on required ports
echo "📡 Freeing ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1

# Create new tmux session with backend window
echo "🐍 Starting Python backend..."
tmux new-session -d -s $SESSION_NAME -n backend -c "$PROJECT_DIR"
tmux send-keys -t $SESSION_NAME:backend "source .venv/bin/activate && python -m krakenagents.server" Enter

# Create frontend window
echo "⚛️  Starting React frontend..."
tmux new-window -t $SESSION_NAME -n frontend -c "$PROJECT_DIR/dashboard"
tmux send-keys -t $SESSION_NAME:frontend "npm run dev" Enter

# Wait for services to start
sleep 3

echo ""
echo "✅ Services started!"
echo ""
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "📺 To view logs: tmux attach -t $SESSION_NAME"
echo "   Switch windows: Ctrl+b then 0 (backend) or 1 (frontend)"
echo "   Detach: Ctrl+b then d"
echo ""
