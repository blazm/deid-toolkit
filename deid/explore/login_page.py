"""Login page — gateway to the DeID toolkit."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.set_page_config(page_title="Login — Face De-Identification Toolkit")
    # Enable browser password autocomplete / password managers
    st.markdown(
        """
        <style>
        input[type="password"] {
            autocomplete: current-password !important;
        }
        input[type="text"] {
            autocomplete: username !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title
    st.title("De-Identification Toolkit Login")
    st.markdown("Only authorized users can access sensitive facial data.")

    # Tabs for login / register / reset
    tab_login, tab_register, tab_reset = st.tabs(["Login", "Create Account", "Reset Password"])

    with tab_login:
        st.subheader("Sign in")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign in", type="primary", use_container_width=True):
            if username and password:
                try:
                    from deid.explore.auth import AuthManager
                    auth = AuthManager()
                    if auth.verify(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.workspace = auth.get_workspace(username)
                        target = st.session_state.pop("pending_page", "toolkit")
                        st.query_params["page"] = target
                        st.session_state.current_page = target
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Auth error: {e}")
            else:
                st.warning("Please enter both username and password.")

    with tab_register:
        st.subheader("Create account")
        new_user = st.text_input("Username", key="reg_username")
        new_pass = st.text_input("Password", type="password", key="reg_password")
        confirm_pass = st.text_input("Confirm password", type="password", key="reg_confirm")

        if st.button("Create account", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Please enter both username and password.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                try:
                    from deid.explore.auth import AuthManager
                    auth = AuthManager()
                    if auth.create_user(new_user, new_pass):
                        st.success(f"Account created for **{new_user}**. Please login.")
                    else:
                        st.warning("Username already exists.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_reset:
        st.subheader("Reset password")
        reset_user = st.text_input("Username", key="reset_username")
        reset_pass = st.text_input("New password", type="password", key="reset_password")
        confirm_reset = st.text_input("Confirm new password", type="password", key="reset_confirm")

        if st.button("Reset password", use_container_width=True):
            if not reset_user or not reset_pass:
                st.warning("Please enter both username and password.")
            elif reset_pass != confirm_reset:
                st.error("Passwords do not match.")
            else:
                try:
                    from deid.explore.auth import AuthManager
                    auth = AuthManager()
                    if auth.reset_password(reset_user, reset_pass):
                        st.success(f"Password reset for **{reset_user}**. Please login.")
                    else:
                        st.warning("User not found.")
                except Exception as e:
                    st.error(f"Error: {e}")
