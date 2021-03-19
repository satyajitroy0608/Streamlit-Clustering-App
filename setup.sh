mkdir -p ~/.streamlit/
echo "[general]
email = \"satyajit12.roy@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
