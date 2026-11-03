"""Start the Stage 10 app on loopback only; not a public deployment."""
import uvicorn

if __name__ == '__main__':
    uvicorn.run('edgemind.api:app', host='127.0.0.1', port=8000, reload=False, log_level='info')
