# Deployment Guide

This guide covers deploying the Face Attendance System.

## Table of Contents

- [Platform Assessment](#platform-assessment)
- [Desktop Application](#desktop-application)
- [API Deployment](#api-deployment)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Post-Deployment](#post-deployment)

## Platform Assessment

### Desktop Application (Local Only)

The desktop application (`codes/ultimate_system.py`) is designed for **local use only**:
- Requires direct webcam access
- Uses Tkinter GUI
- Cannot be deployed to cloud platforms

**Best for**: Classrooms, offices, or any environment with local camera access.

### REST API (Cloud Deployable)

The Flask API (`deployment/api.py`) can be deployed to cloud platforms:
- Accepts image uploads for face registration
- JWT-based authentication
- MySQL database backend

## ⚠️ Why Not Netlify?

Netlify is **NOT suitable** for this project because:

| Requirement | Netlify Limitation |
|------------|-------------------|
| Flask server process | Netlify only supports serverless functions |
| Real-time processing | Serverless functions timeout after 10-26 seconds |
| Persistent connections | No support for long-running processes |
| MySQL database | No built-in database support |

### Recommended Platforms

| Platform | Best For | Pricing |
|----------|----------|---------|
| **Railway** | Easy Flask deployment | Free tier available |
| **Render** | Background workers support | Free tier available |
| **Heroku** | Established PaaS | ~$7/month |
| **DigitalOcean App Platform** | Scalability | ~$5/month |

---

## API Deployment

### Option 1: Railway (Recommended)

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `realtime-face-attendance`

3. **Add MySQL Database**
   - Click "Add Service" → "Database" → "MySQL"
   - Railway will auto-provision and set environment variables

4. **Configure Environment**
   - Go to Variables tab
   - Add these variables:
     ```
     SECRET_KEY=your-super-secret-key
     PORT=5001
     ```

5. **Deploy**
   - Railway auto-deploys on push to main branch
   - Check logs for any issues

### Option 2: Render

1. **Prerequisites**
   - `render.yaml` is already configured in the repo

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub

3. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Render will detect `render.yaml`

4. **Configure Database**
   - Create a managed MySQL database on Render or use external provider
   - Add database credentials as environment variables

5. **Deploy**
   - Click "Create Web Service"
   - Monitor deployment logs

### Option 3: Heroku

1. **Install Heroku CLI**
   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create face-attendance-api
   ```

3. **Add MySQL Add-on**
   ```bash
   heroku addons:create jawsdb:kitefin
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set SECRET_KEY=your-secret-key
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | JWT signing key (generate securely) |
| `DB_HOST` | Yes | MySQL host |
| `DB_USER` | Yes | MySQL username |
| `DB_PASSWORD` | Yes | MySQL password |
| `DB_NAME` | Yes | Database name (default: `face_attendance`) |
| `PORT` | No | Server port (default: 5001) |

### Generating a Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Database Setup

### Initial Schema

The database schema is in `database/init_db.sql`:

```sql
CREATE DATABASE IF NOT EXISTS face_attendance;
USE face_attendance;

CREATE TABLE IF NOT EXISTS Attendance (
    ID INT NOT NULL AUTO_INCREMENT,
    ENROLLMENT VARCHAR(100) NOT NULL,
    NAME VARCHAR(50) NOT NULL,
    DATE VARCHAR(20) NOT NULL,
    TIME VARCHAR(20) NOT NULL,
    SUBJECT VARCHAR(100) NOT NULL,
    PRIMARY KEY (ID)
);
```

### Running Migrations

**Local:**
```bash
mysql -u root -p < database/init_db.sql
```

**Railway:**
```bash
railway run mysql < database/init_db.sql
```

**Heroku (JawsDB):**
```bash
heroku run mysql --host=$JAWSDB_HOST -u$JAWSDB_USERNAME -p$JAWSDB_PASSWORD < database/init_db.sql
```

---

## Post-Deployment

### Verify Deployment

1. **Health Check**
   ```bash
   curl https://your-app-url/api/health
   ```
   
   Expected response:
   ```json
   {"status": "healthy", "timestamp": "2026-01-29T12:00:00"}
   ```

2. **Test Login**
   ```bash
   curl -X POST https://your-app-url/api/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "password"}'
   ```

### Monitoring

- **Logs**: Check platform-specific logs for errors
- **Uptime**: Set up uptime monitoring (UptimeRobot, Pingdom)
- **Performance**: Monitor response times and memory usage

### Security Checklist

- [ ] Changed default `SECRET_KEY`
- [ ] Database password is strong
- [ ] HTTPS enabled (automatic on most platforms)
- [ ] Rate limiting configured (if applicable)
- [ ] CORS origins restricted in production

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Ensure all dependencies in `requirements.txt` |
| Database connection failed | Check DB_* environment variables |
| Port already in use | Change PORT environment variable |
| Camera not found | Desktop app only - cannot use camera in cloud |

### Getting Help

- Open an issue on [GitHub](https://github.com/yxshee/realtime-face-attendance/issues)
- Contact: yash999901@gmail.com
