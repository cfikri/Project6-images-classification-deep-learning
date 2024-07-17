#!/bin/bash

if [ -z "$PORT" ]; then
  PORT=8501
fi

streamlit run api.py --server.port $PORT