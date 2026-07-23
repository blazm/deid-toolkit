"""Simple file-based authentication for the explore app.

Users are stored in a YAML file (default: ``.deid-users.yaml``) with
username and bcrypt password hash. No external database required.

Usage:
    from deid.explore.auth import AuthManager

    auth = AuthManager()
    auth.create_user("alice", "secretpass")
    if auth.verify("alice", "secretpass"):
        print("Logged in")
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import yaml


DEFAULT_USERS_FILE = ".deid-users.yaml"


class AuthManager:
    """Manages user accounts stored in a local YAML file."""

    def __init__(self, users_file: Optional[str] = None) -> None:
        self.users_file = Path(users_file) if users_file else Path(DEFAULT_USERS_FILE)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.users_file.exists():
            self.users_file.write_text(yaml.dump({"users": {}}))

    def _load_users(self) -> dict:
        data = yaml.safe_load(self.users_file.read_text())
        return data or {"users": {}}

    def _save_users(self, data: dict) -> None:
        self.users_file.write_text(yaml.dump(data))

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user with the given password.

        Returns True if created, False if user already exists.
        """
        data = self._load_users()
        if username in data["users"]:
            return False
        # Hash password with bcrypt if available, otherwise use SHA-256
        try:
            import bcrypt
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        except ImportError:
            # Fallback: SHA-256 with salt
            salt = os.urandom(16).hex()
            hashed = f"sha256:{salt}:{hashlib.sha256(password.encode() + salt.encode()).hexdigest()}"
        data["users"][username] = {"password_hash": hashed, "workspace": os.path.expanduser("~")}
        self._save_users(data)
        return True

    def verify(self, username: str, password: str) -> bool:
        """Verify username and password. Returns True if valid."""
        data = self._load_users()
        if username not in data["users"]:
            return False
        hashed = data["users"][username]["password_hash"]
        if hashed.startswith("sha256:"):
            # SHA-256 fallback
            _, salt, expected_hash = hashed.split(":")
            computed = hashlib.sha256(password.encode() + salt.encode()).hexdigest()
            return computed == expected_hash
        else:
            # bcrypt
            try:
                import bcrypt
                return bcrypt.checkpw(password.encode(), hashed.encode())
            except ImportError:
                return False

    def get_workspace(self, username: str) -> str:
        """Get the workspace directory for a user."""
        data = self._load_users()
        if username in data["users"]:
            return data["users"][username].get("workspace", os.path.expanduser("~"))
        return os.path.expanduser("~")

    def set_workspace(self, username: str, workspace: str) -> None:
        """Set the workspace directory for a user."""
        data = self._load_users()
        if username in data["users"]:
            data["users"][username]["workspace"] = workspace
            self._save_users(data)

    def list_users(self) -> list[str]:
        """Return list of usernames."""
        data = self._load_users()
        return list(data["users"].keys())

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns True if deleted, False if not found."""
        data = self._load_users()
        if username in data["users"]:
            del data["users"][username]
            self._save_users(data)
            return True
        return False

    def reset_password(self, username: str, new_password: str) -> bool:
        """Reset a user's password. Returns True if reset, False if user not found."""
        data = self._load_users()
        if username not in data["users"]:
            return False
        try:
            import bcrypt
            hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        except ImportError:
            salt = os.urandom(16).hex()
            hashed = f"sha256:{salt}:{hashlib.sha256(new_password.encode() + salt.encode()).hexdigest()}"
        data["users"][username]["password_hash"] = hashed
        self._save_users(data)
        return True
