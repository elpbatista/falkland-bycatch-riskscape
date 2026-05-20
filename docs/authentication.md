# Dataset Authentication Guide

This project downloads datasets from external providers. Most require valid
credentials before data can be retrieved.

- **NASA PO.DAAC** (MUR SST) – accessed over HTTPS using [Earthdata Login](https://urs.earthdata.nasa.gov/)
- **Copernicus Marine Service (CMEMS)** – accessed through the `copernicusmarine` CLI
- **Copernicus Climate Data Store (CDS / ERA5)** – accessed through the `cdsapi` Python client
- **Global Fishing Watch (GFW)** – accessed with an API token stored outside the repository

---

## 1. NASA Earthdata (PO.DAAC SST)

### 1.1 Create an Earthdata account

Register at:  
<https://urs.earthdata.nasa.gov/>

### 1.2 Configure credentials

The downloader uses `curl` with `-n` to read a `.netrc` file and handles cookies automatically.

#### macOS / Linux

1. Create a file at `~/.netrc` containing:

   ```text
   machine urs.earthdata.nasa.gov
   login YOUR_USERNAME
   password YOUR_PASSWORD
   ```

2. Restrict permissions:

   ```bash
   chmod 600 ~/.netrc
   ```

#### Windows

1. Create `%USERPROFILE%\.netrc` with the same content as above.
2. Ensure it is plain text and located in your user profile directory.

> **Tip:** Do not use a rich-text editor.

### 1.3 Cookie handling

The download script uses:

- `-c <cookie_file>` to write/update cookies  
- `-b <cookie_file>` to send cookies with requests  

Default locations:

| Platform      | Cookie file                  |
|---------------|------------------------------|
| macOS / Linux | `~/.urs_cookies`             |
| Windows       | `%USERPROFILE%\.urs_cookies` |

These files are created automatically by the script.

### 1.4 Verify authentication

**macOS / Linux:**

```sh
curl -n -I https://urs.earthdata.nasa.gov
```

**Windows (PowerShell):**

```powershell
curl.exe -n -I https://urs.earthdata.nasa.gov
```

A successful response returns HTTP 200 or 302.

---

## 2. Copernicus Marine Service (CMEMS)

### 2.1 Create an account

Register at:  
<https://marine.copernicus.eu/>

### 2.2 Install the Copernicus Marine Toolbox

```sh
pip install copernicusmarine
```

### 2.3 Log in

```sh
copernicusmarine login
```

Credentials are stored locally by the CLI.

### 2.4 Test access

```sh
copernicusmarine describe --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D
```

A metadata response confirms authentication.

---

## 3. Copernicus Climate Data Store (CDS / ERA5)

ERA5 wind data is accessed via the CDS API.

### 3.1 Create an account

Register at:  
<https://cds.climate.copernicus.eu/>

Before downloading ERA5 data, you must accept the license terms for the dataset in the CDS web interface.

### 3.2 Install the CDS API client

```sh
pip install cdsapi
```

### 3.3 Configure API credentials

After logging into CDS:

1. Go to **Profile → API key**
2. Copy your credentials
3. Create a file:

   - macOS / Linux: `~/.cdsapirc`
   - Windows: `%USERPROFILE%\.cdsapirc`

File contents:

```text
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Replace with your actual UID and API key.

### 3.4 Restrict permissions (macOS / Linux)

```bash
chmod 600 ~/.cdsapirc
```

### 3.5 Test access

Create a minimal Python test:

```python
import cdsapi
cdsapi.Client()
```

If no authentication error appears, the configuration is correct.

---

## 4. Global Fishing Watch

Global Fishing Watch API access uses a bearer token.

1. Request or create a token through Global Fishing Watch.
2. Copy `.env.example` to `.env`.
3. Set:

   ```text
   GFW_TOKEN=YOUR_TOKEN
   ```

The `.env` file is ignored by Git and must not be committed. Pipeline code
should read this value from the environment rather than from `config.yaml`.

---

## 5. Credential Storage Locations

| Service               | Credential File                  |
|-----------------------|----------------------------------|
| Earthdata             | `~/.netrc`, `~/.urs_cookies`     |
| Copernicus Marine     | Stored by `copernicusmarine` CLI |
| Copernicus CDS (ERA5) | `~/.cdsapirc`                    |
| Global Fishing Watch  | `.env` / `GFW_TOKEN`             |

(Windows equivalents use `%USERPROFILE%`.)

---

## 6. Security Best Practices

- **Never commit credentials to version control**
- Do not store passwords or API keys in:
  - source code
  - `config.yaml`
  - notebooks
  - shared repositories

Recommended approach:

- `.netrc` + script-managed cookies for Earthdata  
- `copernicusmarine login` for CMEMS  
- `.cdsapirc` for CDS (ERA5)  
- `.env` + `GFW_TOKEN` for Global Fishing Watch

Keep these files outside the repository and secured with proper file permissions.
