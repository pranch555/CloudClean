# Accounts

Everyone who uses CloudClean signs in with a username and password. This shows you who uses CloudClean, when, and
who started each job.

Code: `cloudclean/web/auth.py` (accounts, sign-ins, activity), `cloudclean/web/routes_auth.py` (the API) and the
sign-in check in `cloudclean/web/server.py`. Tests: `tests/test_auth.py`.

## What it does

* When someone opens CloudClean and is not signed in, they see the sign-in page. A new visitor clicks **Create
  account**, picks a name and a password, and is signed in straight away.
* **The first account ever created is the admin.** Create your own account first, right after installing or
  updating.
* **Everyone shares the same projects and scans.** Accounts record who did what. They do not hide anyone's work
  from anyone else.
* **What is recorded:**
  * for each account: when it was created, the last sign-in, when it was last seen (updated at most once a
    minute), how many sign-ins, and how many jobs it started
  * an activity list of events: account created, signed in, **failed sign-in** (the account, when the name
    typed is one, and the address it came from), signed out, job started (with its title), password changed,
    account removed, new bridge key. A failed sign-in with a name that is not an account is listed without the
    name, so a password typed into the name box never ends up in the list.
* **Where it is kept:** in the `auth` folder of the workspace. Back it up along with the rest of the workspace.
  * As a service (`deploy/install-service.sh`, e.g. on the Spark): `~/cloudclean-workspace/auth/`.
  * On a PC started with `CloudClean.bat`: `workspace\auth\` inside the CloudClean folder.

  | File | Holds |
  |---|---|
  | `users.json` | the accounts and their password hashes (never the passwords) |
  | `users.json.bak` | `users.json` as it was before CloudClean last changed it: the way back if a hand edit goes wrong |
  | `sessions.json` | who is signed in on which device (only fingerprints of the sign-in cookies) |
  | `activity.jsonl` | the activity list (cut back to the newest 5000 events when it passes 2 MB) |
  | `machine_key` | the bridge key |

## Using it

* **Create account / Sign in.** Names are 3 to 32 letters, digits, dots, dashes or underscores, and start with
  a letter or digit. Upper and lower case count as the same, so `Alice` and `alice` are one account. Passwords
  need at least 8 characters.
* **Settings → Account:** change your password. You need your current password. Your other devices are signed
  out, and this one stays signed in.
* **Settings → Users** (admin only):
  * every account with its created date, last seen, number of sign-ins and jobs, and how many devices are signed
    in
  * the latest 150 activity events
  * **Remove** an account. That person is signed out everywhere at once. You cannot remove your own account.
  * the **bridge key**. You can show it or make a new one.

## The Revo Metro bridge and scripts

The bridge on the scanning PC is not a person, so it uses the **bridge key** instead of a password. Copy the key
from Settings → Users:

```powershell
cloudclean bridge --server http://<spark>:8765 --watch "<folder>" --key <key>
# or: python -m cloudclean.capture.bridge --server http://<spark>:8765 --watch "<folder>" --key <key>
```

You can also store the key once in the `CLOUDCLEAN_KEY` environment variable and leave out `--key`. On Windows,
run `setx CLOUDCLEAN_KEY "<key>"` and then open a new window. If the key is missing or wrong, the bridge stops
with: *The server wants the bridge key: add --key <key>*.

Scripts send the same key in the `X-CloudClean-Key` header:

```bash
curl -H "X-CloudClean-Key: <key>" http://<spark>:8765/api/assets
```

* The key works for every CloudClean API call except the account ones. It cannot see or change accounts, or
  show or change the key itself.
* The live capture view accepts the key too.
* Jobs started with the key are listed under the name **bridge**. The name is kept for this: nobody can create an
  account called `bridge`.
* When you make a new key, the old one stops working at once. Restart the bridge with the new key. Until then the
  bridge keeps retrying and reports *server answered 401: Sign in first*.

## Switching accounts off

`cloudclean serve --no-accounts` starts CloudClean with no sign-in page. Anyone who can reach it can use it, as
before accounts existed. This suits your own PC when nobody else uses it.

Accounts are on by default, and that includes the Spark service. To switch them off there, add `--no-accounts`
to the `ExecStart` line (see `docs/dgx-spark.md`). The `auth` folder is kept and is used again when you switch
accounts back on.

## Security notes

* **Passwords are never stored.** Only a salted scrypt hash is kept.
* **The sign-in cookie holds a random token.** The server keeps only its SHA-256 fingerprint, so a copy of
  `sessions.json` signs nobody in. The cookie is *HttpOnly*, so scripts in the page cannot read it, and
  *SameSite=Lax*.
* **8 wrong passwords in 10 minutes pause that name.** After 8 wrong passwords for one name from one computer
  within 10 minutes, that computer cannot sign in with that name for up to 10 minutes, even with the right
  password.
  * Other names and other computers are not affected.
  * Anyone already signed in stays signed in.
  * Restarting CloudClean also lifts the pause.
* **A sign-in lasts until it has not been used for 30 days.** Each time you use CloudClean (at most once a
  minute), the server and the browser both move the end to 30 days from then, so someone who uses it every day
  stays signed in.
  * Signing out ends that device's sign-in at once.
  * Changing your password ends your sign-ins on your other devices.
  * Removing an account ends all of that person's sign-ins.
* **The API's own pages** (`/docs`, `/redoc`, `/openapi.json`) need a sign-in too.
* **The site uses plain http.** That is fine on your local network and over Tailscale, which encrypts traffic
  between your own devices. Do not open port 8765 to the internet. If CloudClean ever has to be reachable more
  widely, put HTTPS in front of it.
* **Anyone who can open the page can create an account.** Your network (LAN or tailnet) keeps strangers out. The
  accounts tell you who did what.
* **Treat the bridge key like a password.** Anyone who has it can use CloudClean's API. If it leaks, make a new
  one in Settings → Users.

## If the admin forgets the password

First, stop CloudClean. On the Spark, run `systemctl --user stop cloudclean` (and `systemctl --user start
cloudclean` later to start it again). On a PC, close the CloudClean window. Then go to the workspace's `auth`
folder and **make a copy of `users.json`**. Now pick what you need:

* **Start the accounts over.**
  1. Delete or rename `users.json`.
  2. Start CloudClean and create your account straight away. The first account created on an empty
     `users.json` becomes the admin, whoever gets there first.
  3. Everyone else creates their account again. Projects, scans and the activity list are not touched.
* **Keep everyone's accounts.**
  1. Open `users.json` in a text editor. Each account is one block that starts with its name in lower case,
     like `"alice": {`.
  2. Delete the forgotten admin's block, and the comma between it and the next block. If it is the last block,
     delete the comma before it instead.
  3. Start CloudClean and create that account again on the sign-in page. The same name is fine.
  4. Stop CloudClean once more. In the new block, change `"admin": false` to `"admin": true`, then start
     CloudClean again.

  Any account can be made admin this way, and more than one admin is fine.
* **Sign everyone out**, for example after a laptop was lost: delete `sessions.json`. The accounts stay, and
  everyone signs in again.

> **Keep `users.json` valid JSON.** If CloudClean cannot read the file, nobody can sign in and the sign-in page
> says the file is damaged; nothing is overwritten. Fix the mistake, or put back `users.json.bak` (the file as it
> was before CloudClean last changed it). After editing, check the file with `python -m json.tool users.json`
> before you start CloudClean again.

If someone only locked themselves out with wrong passwords, they do not need any of this: wait 10 minutes, or
restart CloudClean.
