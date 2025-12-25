  ---
  Commands om zelf te doen:

  # 1. Maak nieuwe tmux sessie
  tmux new-session -d -s crewai -n backend

  # 2. Start backend in eerste window
  tmux send-keys -t crewai:backend 'cd /Users/charafchnioune/Desktop/Code/tools/crewAI247 && source .venv/bin/activate && python -m krakenagents.server' Enter

  # 3. Maak tweede window voor frontend
  tmux new-window -t crewai -n frontend -c /Users/charafchnioune/Desktop/Code/tools/crewAI247/dashboard

  # 4. Start frontend
  tmux send-keys -t crewai:frontend 'npm run dev' Enter

  # 5. Attach aan sessie om te bekijken
  tmux attach -t crewai

  Navigatie in tmux:
  - Ctrl+b dan 0 = backend window
  - Ctrl+b dan 1 = frontend window
  - Ctrl+b dan d = detach (sessie blijft draaien)

  Als poorten bezet zijn:
  lsof -ti:8000 | xargs kill -9  # Backend port
  lsof -ti:3000 | xargs kill -9  # Frontend port

/Users/charafchnioune/Desktop/Code/tools/crewAI247/start.sh