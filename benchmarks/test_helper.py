import requests

API = "http://localhost:8000"   # your local backend

def get_token(email="test@example.com", password="testpass123"):
    # try to log in; if the account doesn't exist yet, create it
    r = requests.post(f"{API}/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        r = requests.post(f"{API}/auth/signup", json={
            "email": email, "username": "tester",
            "password": password, "gender": "female"})
    r.raise_for_status()
    return r.json()["token"]

if __name__ == "__main__":
    print(get_token())