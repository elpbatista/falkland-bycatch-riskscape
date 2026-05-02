# Dataset Authentication Guide

This project downloads datasets from two external providers. Both require valid credentials before any data can be retrieved.

- **NASA PO.DAAC** (MUR SST) – accessed over HTTPS using [Earthdata Login](https://urs.earthdata.nasa.gov/).
- **Copernicus Marine Service (CMEMS)** – accessed through the `copernicusmarine` CLI.

---

## 1. NASA Earthdata (PO.DAAC SST)

### 1.1 Create an Earthdata account

Register at:  
<https://urs.earthdata.nasa.gov/>

### 1.2 Configure credentials

The downloader uses `curl` with `-n` to read a `.netrc` file, and it handles cookies automatically.

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

1. In PowerShell or any plain-text editor, create `%USERPROFILE%\.netrc` with the same content as above.
2. Ensure the file sits in your user profile directory (typically `C:\Users\<you>`).

> **Tip:** don’t use a rich-text editor; the file must be plain text.

### 1.3 Cookie handling

The download script uses `curl` options to maintain session cookies:

- `-c <cookie_file>` writes/updates cookies
- `-b <cookie_file>` sends cookies with requests

Default locations:

| Platform      | Cookie file                  |
|---------------|------------------------------|
| macOS / Linux | `~/.urs_cookies`             |
| Windows       | `%USERPROFILE%\.urs_cookies` |

You do **not** need to create these files manually – the script does it for you.

### 1.4 Verify your authentication

Run the following command to check the credentials are working:

**macOS / Linux:**

```sh
curl -n -I https://urs.earthdata.nasa.gov
```

**Windows (PowerShell):**

```powershell
curl.exe -n -I https://urs.earthdata.nasa.gov
```

A successful login returns an HTTP 200 or 302 response. Authentication errors indicate invalid credentials.

---

## 2. Copernicus Marine Service (CMEMS)

### 2.1 Create an account

Register at: <https://marine.copernicus.eu/>

### 2.2 Install the Copernicus Marine Toolbox

```sh
pip install copernicusmarine
```

### 2.3 Log in

```sh
copernicusmarine login
```

Credentials are stored locally for later use.

### 2.4 Test access

```sh
copernicusmarine describe --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D
```

A metadata response indicates successful authentication.

---

## 3. Credential storage locations

- **Earthdata:**  
  - macOS / Linux: `~/.netrc`, `~/.urs_cookies`  
  - Windows: `%USERPROFILE%\.netrc`, `%USERPROFILE%\.urs_cookies`
- **Copernicus Marine:**  
  Stored by the `copernicusmarine` CLI under your user profile directory (location varies by OS).

> **Security reminder:**  
> Never commit these files to version control.

---

## 4. Security best practices

- **Do not** put credentials in:
  - source code
  - `config.yaml`
  - any Git repository

- Recommended approaches:
  - use a `.netrc` file + script‑managed cookies for Earthdata/PO.DAAC
  - use `copernicusmarine login` for CMEMS
