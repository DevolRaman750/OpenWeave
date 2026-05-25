# OpenWeave Research Agents - Setup Guide

## 🔴 Errors You Encountered

### Error 1: Langfuse Authentication Error
```
Authentication error: Langfuse client initialized without public_key.
Client will be disabled. Provide a public_key parameter or set 
LANGFUSE_PUBLIC_KEY environment variable.
```

### Error 2: OpenAI/NVIDIA API Key Error
```
openai.OpenAIError: The api_key client option must be set either by 
passing api_key to the client or by setting the OPENAI_API_KEY environment variable
```

---

## 🔍 Root Cause Analysis

### ❌ What Was Wrong

**In the old code:**
```python
# WRONG: Using the API key AS the environment variable name
public_key=os.getenv("pk-lf-09009284-68f2-4857-8c4d-1a049bf33199")
secret_key=os.getenv("sk-lf-faaec043-0ceb-4d0a-a4e2-f9b7a123eca2")
api_key=os.getenv("nvapi-gCmHxbVaihwIQ3Ycekq-sJXEUiChXgdIRoL5...")
```

This code tries to find environment variables named:
- `pk-lf-09009284-68f2-4857-8c4d-1a049bf33199` ❌ (This is the KEY, not the variable name)
- `sk-lf-faaec043-0ceb-4d0a-a4e2-f9b7a123eca2` ❌ (This is the KEY, not the variable name)
- `nvapi-gCmHxbVaihwIQ3Ycekq-...` ❌ (This is the KEY, not the variable name)

Since those environment variables don't exist, `os.getenv()` returns `None`, causing the errors.

### 🔒 Security Issue

**Your API keys were exposed in source code!** This is a critical security vulnerability:
- Anyone with access to the repository can see your API keys
- They could impersonate your account and make unauthorized requests
- This could incur unexpected costs or compromise data

---

## ✅ Solution

### Step 1: Create a `.env` File

Copy `.env.example` to `.env`:
```bash
cd c:\Agents\OpenWeave\openweave-observabily\openweave-research\agents
copy .env.example .env
```

### Step 2: Fill in Your Credentials

Edit the `.env` file with your actual credentials:

```bash
# Add your Langfuse credentials
LANGFUSE_PUBLIC_KEY=pk-lf-your-actual-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-actual-secret-key

# Add your NVIDIA API key
OPENAI_API_KEY=nvapi-your-actual-api-key
```

### Step 3: Get Your Credentials

#### 📍 Langfuse Credentials

**For Local Langfuse:**
1. Open http://localhost:3000 in browser
2. Click **Settings** (top right)
3. Click **API Keys**
4. Copy the **Public Key** and **Secret Key**

**For Langfuse Cloud:**
1. Go to https://cloud.langfuse.com
2. Sign in to your account
3. Click **Settings** → **API Keys**
4. Copy the keys

#### 📍 NVIDIA API Key

1. Go to https://build.nvidia.com/
2. Sign in or create account
3. Navigate to your API Keys section
4. Create or copy your API key (starts with `nvapi-`)

### Step 4: Verify Setup

Test your configuration:
```bash
# Run the test agent
python test_agent.py
```

If successful, you should see:
```
[Agent] Question: What is artificial intelligence?
[Step 1 - Plan] Search query: ...
[Step 2 - Tool] Result: ...
[Step 3 - Answer] ...
```

---

## 📋 Environment Variables Reference

| Variable | Purpose | Example | Required |
|----------|---------|---------|----------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public API key | `pk-lf-...` | ✅ Yes |
| `LANGFUSE_SECRET_KEY` | Langfuse secret API key | `sk-lf-...` | ✅ Yes |
| `LANGFUSE_HOST` | Langfuse server URL | `http://localhost:3000` | ❌ No (default: local) |
| `OPENAI_API_KEY` | NVIDIA API key | `nvapi-...` | ✅ Yes |
| `LANGFUSE_DEBUG` | Enable debug logging | `true` or `false` | ❌ No |
| `LANGFUSE_TRACING_ENVIRONMENT` | Environment name | `development` | ❌ No |

---

## 🔐 Security Best Practices

### ✅ DO

- ✓ Store API keys in `.env` file
- ✓ Add `.env` to `.gitignore` (already done)
- ✓ Use environment variables for all sensitive data
- ✓ Rotate API keys regularly
- ✓ Use least-privilege keys (read-only when possible)

### ❌ DON'T

- ✗ Hardcode API keys in source files
- ✗ Commit `.env` file to Git
- ✗ Share API keys in messages/chats
- ✗ Use API keys as environment variable names
- ✗ Log or print API keys

---

## 🛠️ Troubleshooting

### Problem: "LANGFUSE_PUBLIC_KEY is not set"

**Solution:**
```bash
# Option 1: Set environment variable (temporary)
export LANGFUSE_PUBLIC_KEY=pk-lf-your-key
export LANGFUSE_SECRET_KEY=sk-lf-your-key
export OPENAI_API_KEY=nvapi-your-key

# Option 2: Create .env file (persistent)
# See Step 1-2 above
```

### Problem: "openai.OpenAIError: The api_key client option must be set"

**Solution:**
```bash
# Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY=nvapi-your-key

# Or in .env:
OPENAI_API_KEY=nvapi-your-key
```

### Problem: "Connection refused" to Langfuse

**Solution:**
- Verify Langfuse is running: `docker compose up` in the langfuse directory
- Check Langfuse host is correct:
  - Local: `LANGFUSE_HOST=http://localhost:3000`
  - Cloud: `LANGFUSE_HOST=https://cloud.langfuse.com`

### Problem: "Invalid API key for NVIDIA"

**Solution:**
- Verify key starts with `nvapi-`
- Check key hasn't expired
- Get new key from https://build.nvidia.com/

---

## 📊 Verification Checklist

After setup, verify everything works:

- [ ] `.env` file created in agents directory
- [ ] `.env` file has LANGFUSE_PUBLIC_KEY
- [ ] `.env` file has LANGFUSE_SECRET_KEY
- [ ] `.env` file has OPENAI_API_KEY
- [ ] `.env` file is in `.gitignore`
- [ ] `test_agent.py` runs without credential errors
- [ ] Traces appear in Langfuse dashboard
- [ ] No API keys visible in console output

---

## 📚 What Changed in test_agent.py

| Change | Before | After |
|--------|--------|-------|
| Env var names | `os.getenv("pk-lf-...")` | `os.getenv("LANGFUSE_PUBLIC_KEY")` |
| Error handling | No checks | Validates credentials, shows helpful error messages |
| Security | Keys hardcoded | Keys from environment/`.env` |
| Documentation | Minimal | Comprehensive with setup instructions |

---

## 🎯 Next Steps

1. **Create `.env` file** with your credentials
2. **Run test_agent.py** to verify setup
3. **Check Langfuse dashboard** for traces
4. **Explore trace hierarchy** to understand observability
5. **Reference INSTRUMENTATION_GUIDE.md** for best practices

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** - Most common issues are covered
2. **Enable debug mode** - Set `LANGFUSE_DEBUG=true` in `.env`
3. **Verify credentials** - Ensure keys are correct and not expired
4. **Check services** - Verify Langfuse/NVIDIA services are reachable
5. **Review logs** - Error messages will guide you to the issue

---

## 🔗 References

- [Langfuse Documentation](https://langfuse.com/docs)
- [NVIDIA Build API Keys](https://build.nvidia.com/)
- [Environment Variables Guide](../TRACING_CONFIGURATION.md)
- [Instrumentation Best Practices](../INSTRUMENTATION_GUIDE.md)

---

**Setup Date**: May 15, 2026  
**Status**: ✅ Complete
