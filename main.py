from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Hello World"}