import pytest
from fastapi.testclient import TestClient

def test_register_user(client: TestClient):
    # Test valid registration
    response = client.post(
        "/api/auth/register",
        json={"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

def test_register_duplicate_email(client: TestClient):
    # Register first user
    client.post(
        "/api/auth/register",
        json={"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}
    )
    # Register duplicate user
    response = client.post(
        "/api/auth/register",
        json={"username": "anotheruser", "email": "testuser@example.com", "password": "anotherpassword"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_login_user(client: TestClient):
    # Register user
    client.post(
        "/api/auth/register",
        json={"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}
    )
    # Login
    response = client.post(
        "/api/auth/login",
        json={"email": "testuser@example.com", "password": "securepassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data

def test_login_invalid_password(client: TestClient):
    client.post(
        "/api/auth/register",
        json={"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "testuser@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert "Incorrect email" in response.json()["detail"]

def test_get_current_user_profile(client: TestClient):
    # Register
    reg_response = client.post(
        "/api/auth/register",
        json={"username": "testuser", "email": "testuser@example.com", "password": "securepassword"}
    )
    access_token = reg_response.json()["access_token"]

    # Request profile
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "testuser@example.com"
    assert "id" in data
