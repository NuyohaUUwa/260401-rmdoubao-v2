function qs(id) {
  return document.getElementById(id);
}

function setAuthMessage(message, isError = false) {
  const el = qs("auth-message");
  el.textContent = message;
  el.className = isError ? "submit-message error-text" : "submit-message";
}

function switchTab(mode) {
  const loginForm = qs("login-form");
  const registerForm = qs("register-form");
  const loginBtn = qs("tab-login");
  const registerBtn = qs("tab-register");
  if (mode === "register") {
    loginForm.classList.add("hidden");
    registerForm.classList.remove("hidden");
    loginBtn.classList.remove("active");
    registerBtn.classList.add("active");
    setAuthMessage("");
  } else {
    registerForm.classList.add("hidden");
    loginForm.classList.remove("hidden");
    registerBtn.classList.remove("active");
    loginBtn.classList.add("active");
    setAuthMessage("");
  }
}

async function submitLogin(event) {
  event.preventDefault();
  const formData = new FormData();
  formData.append("username", qs("login-username").value.trim());
  formData.append("password", qs("login-password").value);
  try {
    const resp = await fetch("/api/auth/login", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "登录失败");
    }
    setAuthMessage("登录成功，正在进入系统...");
    window.location.href = "/";
  } catch (error) {
    setAuthMessage(error.message || "登录失败", true);
  }
}

async function submitRegister(event) {
  event.preventDefault();
  const formData = new FormData();
  formData.append("username", qs("register-username").value.trim());
  formData.append("password", qs("register-password").value);
  try {
    const resp = await fetch("/api/auth/register", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "注册失败");
    }
    qs("login-username").value = data.user.username;
    qs("login-password").value = "";
    setAuthMessage("注册成功，请使用新账号登录。");
    switchTab("login");
  } catch (error) {
    setAuthMessage(error.message || "注册失败", true);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  qs("tab-login").addEventListener("click", () => switchTab("login"));
  qs("tab-register").addEventListener("click", () => switchTab("register"));
  qs("login-form").addEventListener("submit", submitLogin);
  qs("register-form").addEventListener("submit", submitRegister);

  try {
    const meResp = await fetch("/api/auth/me");
    if (meResp.ok) {
      window.location.href = "/";
    }
  } catch (_) {
    // ignore
  }
});
