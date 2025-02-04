from app import app  # Import your Flask app instance

# Vercel needs the "app" variable for deployment
if __name__ == "__main__":
    app.run()
