"""App settings, read from environment variables so no secrets live in the code.

    export WEATHER_API_KEY=your_openweathermap_key   # free key from openweathermap.org
    export SECRET_KEY=any-long-random-string         # keeps logins valid across restarts
"""

import os

weather_api_key = os.environ.get("WEATHER_API_KEY", "")
secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()
database_url = os.environ.get("DATABASE_URL", "sqlite:///database.db")
debug = os.environ.get("FLASK_DEBUG", "0") == "1"
