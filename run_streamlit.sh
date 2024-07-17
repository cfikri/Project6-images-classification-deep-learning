#!/bin/bash

if [ -z "$PORT" ]; then
  PORT=8501
fi

streamlit run app.py --server.port $PORT
