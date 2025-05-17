from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    pass

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
