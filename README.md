# kos-insights-llm-agent
python micro service for LLM, audio call transcription


### Docker Instructions
<br>
build the docker image with `sudo docker build -t llm-agent .`

<br>
run the docker image with `sudo docker run --env-file .env -p 5000:5000 llm-agent`

<br>
can also run by `sudo docker compose up`
