# Step-by-Step Guide: Moving Your Portfolio Dashboard Online

This guide explains how to host your dashboard online so you can share it as a link instead of a ZIP file.

## Step 1: Upload to GitHub (Free)
1. Create a free account on [GitHub](https://github.com).
2. Create a new "Repository" (e.g., `portfolio-dashboard`).
3. Upload **all** files from your folder to this repository.

## Step 2: Host the Backend (API)
We recommend **Render.com** (it has a free tier):
1. Sign up for [Render](https://render.com).
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Use these settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py` (or `uvicorn main:app --host 0.0.0.1 --port $PORT`)
5. Once deployed, Render will give you a URL (e.g., `https://portfolio-backend.onrender.com`). **Copy this URL.**

## Step 3: Link the Frontend to the Backend
1. Open `index.html` in your GitHub repository.
2. Find the line: `const PRODUCTION_API_URL = "";` (around line 553).
3. Paste your Render URL inside the quotes:
   ```javascript
   const PRODUCTION_API_URL = "https://portfolio-backend.onrender.com";
   ```
4. Save (Commit) the changes in GitHub.

## Step 4: Host the Frontend (The Dashboard)
Still on GitHub:
1. Go to your repository **Settings** > **Pages**.
2. Under "Build and deployment", set the source to **Deploy from a branch**.
3. Select the `main` branch and `/root` folder, then click **Save**.
4. GitHub will give you a final link (e.g., `https://your-username.github.io/portfolio-dashboard/`).

**That's it!** You can now share this final link with anyone. They don't need to install Python or run any `.bat` files.

---
### 💡 Important Note
The free tier of Render "goes to sleep" after 15 minutes of inactivity. The first time someone opens your link, it might take 30-60 seconds for the "Backend Online" status to appear while the server wakes up.
